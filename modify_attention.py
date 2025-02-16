import os
import torch

ASCENDING = int(os.environ.get("ASCENDING", True))!=0 #* ASCENDING
REPLACE_WITH_SCATTERED_ATTN = int(os.environ.get("REPLACE_WITH_SCATTERED_ATTN", False))!=0 #* 是否用 scattered mask 来替换 local mask 来进行 mask
MASK_ATTN = int(os.environ.get("MASK_ATTN", False))!=0 #* 默认不对 uniform attn 进行掩码
MASK_TYPE = int(os.environ.get("MASK_TYPE", 0)) #* 确定要掩码的 attention 类型 - 0 表示小rk, 1 表示rk中间的，2表示大rk，默认屏蔽中间的 attention


def transform_ratio(ratio, epsilon=1e-6):
    return torch.where((ratio<1)*(ratio>epsilon), 1.0 / ratio, ratio)


def attention_distrituion_equilibrium_rank(multihead_attentions, use_naive_mean=True, epsilon=1e-6):
     #* shape = [num_heads, n_tks/1, n_tks]
    visible_num_tokens = multihead_attentions.size(1)
    num_tokens = multihead_attentions.size(2)
    device = multihead_attentions.device
    if visible_num_tokens>1: target_attentions = 1/torch.arange(1,num_tokens+1).to(device).reshape(1, -1, 1)
    else: target_attentions = torch.Tensor([1/num_tokens]).to(device).unsqueeze(0).unsqueeze(0)
    #* [num_heads, n_tks/1, n_tks]
    ratio = transform_ratio(multihead_attentions/target_attentions, epsilon=epsilon)
    scores = torch.sum(torch.sum(ratio, dim=-1), dim=-1)
    if visible_num_tokens>1:
        scores /= (num_tokens*(num_tokens+1))>>1
    else:
        scores /= num_tokens
    return scores
    


def rank_attention_heads(multihead_attentions, topk, indicator_rank=True, alpha=5.0, use_naive_mean=True, replace_repeatedly=True):
    
    num_heads = multihead_attentions.size(0)
    mid_idx = num_heads >> 1
    scores = attention_distrituion_equilibrium_rank(multihead_attentions, use_naive_mean)
    sorted_indices = torch.argsort(scores, descending=False)  
 
    
    if MASK_ATTN:
        mask_attn = torch.zeros(multihead_attentions.shape[-2:]).unsqueeze(0).to(multihead_attentions.device)
        if MASK_TYPE==0:
            multihead_attentions[sorted_indices[:topk]] = mask_attn
        elif MASK_TYPE==1:
            left_start, right_end = mid_idx-(topk>>1), mid_idx+(topk>>1)+1
            multihead_attentions[sorted_indices[left_start:right_end]] = mask_attn
        else: multihead_attentions[sorted_indices[-topk:]] = mask_attn
        return multihead_attentions
    
    if replace_repeatedly:
        if ASCENDING:
            multihead_attentions[sorted_indices[-topk:]] = multihead_attentions[sorted_indices[:1]]
        else:
            if REPLACE_WITH_SCATTERED_ATTN: multihead_attentions[sorted_indices[:topk]] = multihead_attentions[sorted_indices[mid_idx:mid_idx+1]]
            else: multihead_attentions[sorted_indices[:topk]] = multihead_attentions[sorted_indices[-1:]]
    else:
        if ASCENDING: multihead_attentions[sorted_indices[-topk:]] = torch.mean(multihead_attentions[sorted_indices[:topk]], dim=0, keepdim=True)
        else:
            if REPLACE_WITH_SCATTERED_ATTN: 
                left_start, right_end = mid_idx-(topk>>1), mid_idx+(topk>>1)+1
                multihead_attentions[sorted_indices[:topk]] = torch.mean(multihead_attentions[sorted_indices[left_start:right_end]], dim=0, keepdim=True)
            else: multihead_attentions[sorted_indices[:topk]] = torch.mean(multihead_attentions[sorted_indices[-topk:]], dim=0, keepdim=True)
    return multihead_attentions

