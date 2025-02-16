import os, sys, warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modify_attention import rank_attention_heads



# MODEL_NAME = os.environ.get("MODEL_NAME", "") 
SHALLOW_LAYER = int(os.environ.get("SHALLOW_LAYER", 1)) 
TOPK_RATIO = float(os.environ.get("TOPK_RATIO", 0.1))
REPEAT = int(os.environ.get("REPEAT", True))!=0 
HEAD_CHANGE = int(os.environ.get("HEAD_CHANGE", False))!=0 
ASCENDING = int(os.environ.get("ASCENDING", True))!=0 

from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
    MistralModel,
    MistralDecoderLayer, 
    MistralAttention,
    MistralFlashAttention2,
    MistralSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.cache_utils import Cache
from transformers.utils import (
    logging,
)


logger = logging.get_logger(__name__)


class MistralAttentionWithHeadAdaptation(MistralAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
                multihead_attentions=attn_weights[0],
                topk=round(attn_weights[0].size(0) * TOPK_RATIO) if TOPK_RATIO<=0.99 else int(TOPK_RATIO),
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

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value 


MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttentionWithHeadAdaptation,
    "flash_attention_2": MistralFlashAttention2,
    "sdpa": MistralSdpaAttention,
}

class MistralDecoderLayerWithHeadAdaptation(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

class MistralModelWithHeadAdaptation(MistralModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MistralDecoderLayerWithHeadAdaptation(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class MistralForCausalLMWithHeadAdaptation(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModelWithHeadAdaptation(config)

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
        