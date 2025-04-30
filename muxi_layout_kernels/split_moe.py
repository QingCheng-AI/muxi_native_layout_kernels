from typing import Callable, List, Optional
import torch
from utils import routing_gate, experts_compute


# In vllm deepseek-v3, they split fused_moe into two functions: grouped_topk and fused_experts
# The fused_moe function is just a test function that calls these two functions

def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 scoring_func: str = "softmax",
                 e_score_correction_bias: Optional[torch.Tensor] = None):
    
    assert scoring_func == "softmax" or scoring_func == "sigmoid", "Only softmax and sigmoid are supported now"
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if num_expert_group is None:
        num_expert_group = 1
    if topk_group is None:
        topk_group = 1
    
    B, H = hidden_states.shape
    num_experts = gating_output.shape[1]

    padding_size = 0
    if B % 16 != 0:
        padding_size = 16 - B % 16
        gating_output = torch.cat(
            [
                gating_output,
                torch.zeros(
                    padding_size,
                    num_experts,
                    dtype=gating_output.dtype,
                    device=gating_output.device,
                ),
            ],
            dim=0,
        )
        B = gating_output.size(0)

    expertsIds = torch.empty(B, topk, dtype=torch.int32, device="cpu")
    selected_experts_weights = torch.empty(
        B, topk, dtype=hidden_states.dtype, device="cpu"
    )

    score_fun = 0
    if scoring_func == "softmax":
        score_fun = 0
    elif scoring_func == "sigmoid":
        score_fun = 1
    else:
        raise ValueError("Unsupported scoring function")
    
    routing_gate(
        gating_output,
        score_fun,
        B,
        H,
        num_expert_group,
        topk_group,
        expertsIds,
        selected_experts_weights,
        topk,
        load_balance_bias=e_score_correction_bias
    )

    if renormalize:
        selected_experts_weights = (
            selected_experts_weights
            / selected_experts_weights.sum(dim=-1, keepdim=True)
        )
    
    return selected_experts_weights, expertsIds


def fused_experts(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  inplace: bool = False,
                  use_fp8_w8a8: bool = False,
                  use_int8_w8a16: bool = False,
                  w1_scale: Optional[torch.Tensor] = None,
                  w2_scale: Optional[torch.Tensor] = None,
                  a1_scale: Optional[torch.Tensor] = None,
                  a2_scale: Optional[torch.Tensor] = None,
                  block_shape: Optional[List[int]] = None):

    assert inplace == True, "Only inplace is supported for now"
    assert topk_weights.shape == topk_ids.shape
    assert topk_weights.shape[0] % 16 == 0

    shape = hidden_states.size()
    B, H = hidden_states.shape
    padding_size = 0
    if B % 16 != 0:
        padding_size = 16 - B % 16
        hidden_states = torch.cat(
            [
                hidden_states,
                torch.zeros(
                    padding_size,
                    H,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                ),
            ],
            dim=0,
        )
        B = hidden_states.size(0)
        
    # 2. Compute the experts output
    e1, m1, k1 = w1.shape
    e2, m2, k2 = w2.shape
    assert e1 == e2
    assert k1 == m2

    experts_compute(
        w1,
        w2,
        hidden_states,
        B,
        e1,
        topk_ids,
        topk_weights,
        topk_ids.shape[-1]
    )
    
    if padding_size > 0:
        hidden_states = hidden_states[:-padding_size]
    return hidden_states.view(shape)