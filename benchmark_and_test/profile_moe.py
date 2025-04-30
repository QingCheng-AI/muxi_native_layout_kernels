# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest test_muxi_moe.py`.
"""
import time
import pytest
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from muxi_moe_planC import fused_moe
from utils import fused_moe as iterative_moe

def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    score_func: str,
    dtype: torch.dtype,
):
    a = torch.rand((m, k), device="cuda", dtype=dtype) / 10
    a1 = a.clone()
    w1 = torch.rand((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.rand((e, k, n), device="cuda", dtype=dtype) / 10
    w1_transposed = w1.clone()
    w2_transposed = w2.clone()
    score = torch.rand((m, e), device="cuda", dtype=dtype)
    # score = torch.ones((m, e), device="cuda", dtype=dtype)
    # bias = torch.rand((e), device="cuda", dtype=dtype)
    bias = None

    # layoutA
    for i in range(e):
        w1_shaped = w1_transposed[i].view(2 * n // 16, 16, k // 8, 8)
        w1_transposed[i] = w1_shaped.permute(0, 2, 1, 3).contiguous().view(2 * n, k).contiguous()
        w2_shaped = w2_transposed[i].view(k // 16, 16, n // 8, 8)
        w2_transposed[i] = w2_shaped.permute(0, 2, 1, 3).contiguous().view(k, n).contiguous()


    # maca_output = fused_moe(a, w1_transposed, w2_transposed, score, topk, renormalize=False)

    # iterative_output = iterative_moe(a1, w1, w2, score, topk, global_num_experts=e, renormalize=False)

    maca_output = fused_moe(a, w1_transposed, w2_transposed, score, topk, renormalize=False, use_grouped_topk=True, num_expert_group=8, topk_group=4, gating_bias=bias, score_func=score_func)

    iterative_output = iterative_moe(a1, w1, w2, score, topk, 8, 4, global_num_experts=e, renormalize=False, e_score_correction_bias=bias, score_func=score_func)    

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_base:
        with record_function("iterative_moe"):
            with torch.no_grad():
                iterative_output = iterative_moe(a1, w1, w2, score, topk, 8, 4, global_num_experts=e, renormalize=False, e_score_correction_bias=bias, score_func=score_func)
    
    print(prof_base.key_averages().table(sort_by="cuda_time_total"))    

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("fused_moe"):
            with torch.no_grad():
                maca_output = fused_moe(a1, w1_transposed, w2_transposed, score, topk, renormalize=False, use_grouped_topk=True, num_expert_group=8, topk_group=4, gating_bias=bias, score_func=score_func)
    
    print(prof.key_averages().table(sort_by="cuda_time_total"))    

    # print(maca_output)
    # print(iterative_output)

    torch.testing.assert_close(maca_output, iterative_output, atol=0.1, rtol=0.05)
    print("Test passed")

if __name__ == "__main__":
    test_fused_moe(16, 256, 7168, 256, 8, "softmax", torch.float16)
    test_fused_moe(16, 256, 7168, 256, 8, "sigmoid", torch.float16)