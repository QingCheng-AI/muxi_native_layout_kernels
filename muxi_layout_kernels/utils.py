from typing import Callable, List, Optional
import torch
import muxi_layout_kernels
import torch.nn.functional as F


def scale_whole_tensor(weight, scale):
    e, m, k = weight.shape
    result = weight.clone().to(torch.bfloat16)
    for expert in range(e):
        result[expert] = scale_tensor(result[expert], scale[expert])
    return result

def scale_tensor(weight, scale):
    m, k = weight.shape
    result = weight.clone()
    for i in range(0, m, 128):
        for j in range(0, k, 128):
            # 找到 scale 矩阵中对应的元素
            scale_i = i // 128
            scale_j = j // 128
            scale_factor = scale[scale_i, scale_j]
            # 对 weight 中 (128, 128) 方块进行缩放
            result[i:i + 128, j:j + 128] *= scale_factor
    return result


def routing_gate(
    gating_output: torch.Tensor,
    score_fun: int,
    batch_size: int,
    hidden_size: int,
    n_groups: int,
    topK_groups: int,
    expertsIds: torch.Tensor,
    selected_experts_weights: torch.Tensor,
    topk: int,
    load_balance_bias: Optional[torch.Tensor] = None,
) -> None:
    muxi_layout_kernels.routing_gate(
        gating_output,
        score_fun,
        batch_size,
        hidden_size,
        n_groups,
        topK_groups,
        expertsIds,
        selected_experts_weights,
        topk,
        load_balance_bias,
    )


def experts_compute(
    w1: torch.Tensor,
    w2: torch.Tensor,
    hidden_states: torch.Tensor,
    batch_size: int,
    num_experts: int,
    expertsIds: torch.Tensor,
    selected_experts_weights: torch.Tensor,
    dynamicExpertsPerAct: int = 8,
    bias: Optional[torch.Tensor] = None,
) -> None:
    muxi_layout_kernels.experts_compute(
        w1,
        w2,
        hidden_states,
        batch_size,
        num_experts,
        dynamicExpertsPerAct,
        expertsIds,
        selected_experts_weights,
        bias
    )

# def fused_routing_gate(
#     gating_output: torch.Tensor,
#     score_fun: int,
#     batch_size: int,
#     hidden_size: int,
#     n_groups: int,
#     topK_groups: int,
#     expertsIds: torch.Tensor,
#     selected_experts_weights: torch.Tensor,
#     topk: int,
#     load_balance_bias: Optional[torch.Tensor] = None,
# ) -> None:
#     muxi_layout_kernels.fused_routing_gate(
#         gating_output,
#         score_fun,
#         batch_size,
#         hidden_size,
#         n_groups,
#         topK_groups,
#         expertsIds,
#         selected_experts_weights,
#         topk,
#         load_balance_bias
#     )


# def fused_experts_compute(
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     hidden_states: torch.Tensor,
#     batch_size: int,
#     num_experts: int,
#     expertsIds: torch.Tensor,
#     selected_experts_weights: torch.Tensor,
#     dynamicExpertsPerAct: int = 8,
# ) -> None:
#     muxi_layout_kernels.fused_experts_compute(
#         w1,
#         w2,
#         hidden_states,
#         batch_size,
#         num_experts,
#         dynamicExpertsPerAct,
#         expertsIds,
#         selected_experts_weights
#     )


####################################################################################################
####################################################################################################
####################################################################################################

# torch implementation of fused_moe

def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int = 0,
    topk_group: int = 1,
    global_num_experts: int = 1,
    expert_map: torch.Tensor = None,
    renormalize: bool = False,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    score_func: str = "softmax"
) -> torch.Tensor:
    """
    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
        expert_map: [num_experts]
    """
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = hidden_states.dtype

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, global_num_experts)
    # topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    if score_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1, dtype=dtype)
    elif score_func == "sigmoid":
        scores = torch.sigmoid(gating_output)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
    # topk_weights, selected_experts = topk_weights.topk(topk, dim=-1)

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        group_scores = scores.view(num_token, num_expert_group,
                               -1).topk(2, dim=-1)[0].sum(dim=-1)
    else:
        group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]

    # print(group_scores.shape)

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]

    # print("baseline Group Experts compute done")
    # print(group_scores)
    # print(group_idx)

    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
        topk_ids_sorted = torch.sort(topk_ids, dim=-1, descending=False).values
        # print(topk_weights)
        topk_weights_sorted = torch.sort(topk_weights, dim=-1, descending=False).values
        # print(topk_weights_sorted)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)
        topk_ids_sorted = torch.sort(topk_ids, dim=-1, descending=False).values
        topk_weights_sorted = torch.sort(topk_weights, dim=-1, descending=False).values

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # print("baseline Experts compute done")
    # print(topk_ids_sorted)
    # print(topk_weights.shape)
    # print(topk_weights)
    # print(topk_weights_sorted)
    # print(topk_weights)

    if expert_map is not None:
        topk_ids = expert_map[topk_ids]

    final_hidden_states = None
    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_mask = (topk_ids == expert_idx)
        expert_weights = (topk_weights * expert_mask).sum(dim=-1, keepdim=True)
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[:, :intermediate_size])
        x = x[:, intermediate_size:] * gate
        x = F.linear(x, expert_w2)
        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states

    return final_hidden_states.view(orig_shape)  # type: ignore
