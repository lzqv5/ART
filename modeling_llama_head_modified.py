import os, sys, warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modify_attention import rank_attention_heads

# MODEL_NAME = os.environ.get("MODEL_NAME", "") 
SHALLOW_LAYER = int(os.environ.get("SHALLOW_LAYER", 1)) 
TOPK_RATIO = float(os.environ.get("TOPK_RATIO", 0.1)) 
REPEAT = int(os.environ.get("REPEAT", True))!=0 
HEAD_CHANGE = int(os.environ.get("HEAD_CHANGE", False))!=0 
ASCENDING = int(os.environ.get("ASCENDING", True))!=0 


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
)

logger = logging.get_logger(__name__)
        

class LlamaAttentionWithHeadAdaptation(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
    
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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
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
                #* multihead_attentions.shape = [num_heads, n_tks, n_tks]
                #* attn_weights.shape = [bsz, num_heads, num_tokens, num_tokens]
                multihead_attentions=attn_weights[0],
                topk=round(attn_weights[0].size(0) * TOPK_RATIO) if TOPK_RATIO<=0.999 else int(TOPK_RATIO),
                indicator_rank=False,
                use_naive_mean=False,
                replace_repeatedly=REPEAT,  
            ) #* 修改 attn_weights
        ############## END:MODIFICATION ##############
        
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
        

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttentionWithHeadAdaptation,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}

class LlamaDecoderLayerWithHeadAdaptation(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
    

class LlamaModelWithHeadAdaptation(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerWithHeadAdaptation(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class LlamaForCausalLMWithHeadAdaptation(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModelWithHeadAdaptation(config)

def prepare_input(tokenizer, prompts, device):
    input_tokens = tokenizer.batch_encode_plus(prompts, 
                                               return_tensors="pt", 
                                               padding=True,
                                               add_special_tokens=False,
                                               )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)
    #* input_tokens - <class 'transformers.tokenization_utils_base.BatchEncoding'>
    return input_tokens


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_gen_len):
    #* input_ids.shape = [bsz, seq_len] = [1, seq_len]
    input_ids = tokenizer(prompt, return_tensors="pt",padding=True,add_special_tokens=False).input_ids.to(model.device)
    outputs = model(input_ids=input_ids) #* outputs.keys = odict_keys(['logits', 'past_key_values'])
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1) #* shape=[1,1]
    input_ids = torch.concat([input_ids, pred_token_idx], dim=1)
    for _ in range(max_gen_len-1):
        outputs = model(input_ids=input_ids)
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        input_ids = torch.concat([input_ids, pred_token_idx], dim=1)
        if pred_token_idx == tokenizer.eos_token_id: break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)