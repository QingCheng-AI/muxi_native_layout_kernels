from typing import Callable, List, Optional
import torch
import muxi_layout_kernels


def fused_moe(
    hidden_states: torch.Tensor,  # [batch_size, hidden_size]
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,  # [batch_size, num_experts]
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    gating_bias: Optional[torch.Tensor] = None,
    score_func: str = "softmax",
    soft_fp8: bool = False,
) -> torch.Tensor:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    if num_expert_group is None:
        num_expert_group = 1
    if topk_group is None:
        topk_group = 1
    if not use_grouped_topk and num_expert_group != 1:
        raise ValueError("When use_grouped_topk is False, num_expert_group must be 1.")

    shape = hidden_states.size()
    B, H = hidden_states.shape
    num_experts = gating_output.shape[1]

    # 1. Compute the gating output
    topk_ids = torch.empty(B, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights = torch.empty(
        B, topk, dtype=hidden_states.dtype, device=hidden_states.device
    )

    if score_func == "softmax":
        score_fun = 0
    elif score_func == "sigmoid":
        score_fun = 1
    else:
        raise ValueError("Unsupported scoring function")

    muxi_layout_kernels.fused_routing_gate(
        gating_output,
        score_fun,
        B,
        H,
        num_expert_group,
        topk_group,
        topk_ids,
        topk_weights,
        topk,
        gating_bias,
    )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # 2. Compute the experts output
    e1, m1, k1 = w1.shape
    e2, m2, k2 = w2.shape
    assert e1 == e2
    assert k1 == m2

    micro_batchsize = 16

    # print("begin experts compute")
    topK = topk_weights.size(1)
    max_num_tokens_padded = (topK * B) + e1 * (micro_batchsize - 1)
    sorted_token_ids = torch.empty(
        max_num_tokens_padded, dtype=torch.int32, device="cuda"
    )
    cumsum_buffer = torch.empty(e1 + 1, dtype=torch.int32, device="cuda")
    padded_num_experts = torch.empty(1, dtype=torch.int32, device="cuda")
    experts_ids = torch.empty(
        (max_num_tokens_padded + micro_batchsize - 1) // micro_batchsize,
        dtype=torch.int32,
        device="cuda",
    )
    C = torch.zeros(topK * m1 * B, dtype=hidden_states.dtype, device="cuda")
    y = torch.zeros_like(hidden_states)

    if soft_fp8:
        assert w1_scale is not None and w2_scale is not None
        muxi_layout_kernels.fused_experts_compute(
            w1,
            w2,
            hidden_states,
            B,
            e1,
            topk_ids.shape[-1],
            topk_ids,
            topk_weights,
            sorted_token_ids,
            cumsum_buffer,
            padded_num_experts,
            experts_ids,
            C,
            y,
            w1_scale,
            w2_scale,
            block_shape,
            soft_fp8,
        )
    else:
        muxi_layout_kernels.fused_experts_compute(
            w1,
            w2,
            hidden_states,
            B,
            e1,
            topk_ids.shape[-1],
            topk_ids,
            topk_weights,
            sorted_token_ids,
            cumsum_buffer,
            padded_num_experts,
            experts_ids,
            C,
            y,
        )
    del sorted_token_ids, cumsum_buffer, padded_num_experts, experts_ids, C

    return y.view(shape)
