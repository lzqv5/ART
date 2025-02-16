import os, sys, warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modify_attention import rank_attention_heads

SHALLOW_LAYER = int(os.environ.get("SHALLOW_LAYER", 1)) #* 可以人为指定输入的该模型它在处理对应任务时哪些部分属于 shallow layer
TOPK_RATIO = float(os.environ.get("TOPK_RATIO", 0.1)) #* 人为指定替换多少个 head
REPEAT = int(os.environ.get("REPEAT", True))!=0 #* 是否用最佳的那个 head 来替换所有的 head
HEAD_CHANGE = int(os.environ.get("HEAD_CHANGE", False))!=0 #* 是否启用 head_replace
ASCENDING = int(os.environ.get("ASCENDING", True))!=0 #* ASCENDING

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2DecoderLayer, 
    Qwen2Attention,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.cache_utils import Cache
from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)

class Qwen2AttentionWithHeadAdaptation(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        ############## BEGIN:MODIFICATION ##############
        if HEAD_CHANGE and self.layer_idx < SHALLOW_LAYER:
            attn_weights[0] = rank_attention_heads(
                multihead_attentions=attn_weights[0], # TOPK_RATIO 也即 k
                topk=round(attn_weights[0].size(0) * TOPK_RATIO) if TOPK_RATIO<=0.999 else int(TOPK_RATIO),
                indicator_rank=False,
                use_naive_mean=False,
                replace_repeatedly=REPEAT,  
            ) 
        ############## END:MODIFICATION ##############
        
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2AttentionWithHeadAdaptation,
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}

class Qwen2DecoderLayerWithHeadAdaptation(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        # For distributed inference
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class Qwen2ModelWithHeadAdaptation(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayerWithHeadAdaptation(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class Qwen2ForCausalLMWithHeadAdaptation(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2ModelWithHeadAdaptation(config)

def prepare_input(tokenizer, prompts, device):
    input_tokens = tokenizer.batch_encode_plus(prompts, 
                                               return_tensors="pt", 
                                               padding=True,
                                               add_special_tokens=False,
                                               )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)
    return input_tokens
        