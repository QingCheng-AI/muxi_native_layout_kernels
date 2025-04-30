# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest test_muxi_moe.py`.
"""
import time
import pytest
import torch 

from muxi_layout_kernels.utils import fused_moe as iterative_moe
from muxi_layout_kernels.muxi_moe_planC import fused_moe as planC_moe

def benchmark_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    score_func: str,
    dtype: torch.dtype,
):
    a = torch.rand((m, k), device="cuda", dtype=dtype) / 100
    a1 = a.clone()
    w1 = torch.rand((e, 2 * n, k), device="cuda", dtype=dtype) / 100
    w2 = torch.rand((e, k, n), device="cuda", dtype=dtype) / 100
    w1_transposed = w1.clone()
    w2_transposed = w2.clone()
    score = torch.rand((m, e), device="cuda", dtype=dtype)
    # bias = torch.rand((e), device="cuda", dtype=dtype)
    bias = None

    # layout weights for the MACA operation
    for i in range(e):
        w1_shaped = w1_transposed[i].view(2 * n // 16, 16, k // 8, 8)
        w1_transposed[i] = w1_shaped.permute(0, 2, 1, 3).contiguous().view(2 * n, k).contiguous()
        w2_shaped = w2_transposed[i].view(k // 16, 16, n // 8, 8)
        w2_transposed[i] = w2_shaped.permute(0, 2, 1, 3).contiguous().view(k, n).contiguous()


    for _ in range(100):
        maca_output = planC_moe(a, w1_transposed, w2_transposed, score, topk, renormalize=False, use_grouped_topk=True, num_expert_group=8, topk_group=4, gating_bias=bias, score_func=score_func)
    torch.cuda.synchronize()

    t1 = time.perf_counter()
    for _ in range(100):
        maca_output = planC_moe(a, w1_transposed, w2_transposed, score, topk, renormalize=False, use_grouped_topk=True, num_expert_group=8, topk_group=4, gating_bias=bias, score_func=score_func)
    
    torch.cuda.synchronize()
    t2 = time.perf_counter()


    for _ in range(100):
        iterative_output = iterative_moe(a1, w1, w2, score, topk, 8, 4, global_num_experts=e, renormalize=False, e_score_correction_bias=bias, score_func=score_func)
    torch.cuda.synchronize()

    t3 = time.perf_counter()
    for _ in range(100):
        iterative_output = iterative_moe(a1, w1, w2, score, topk, 8, 4, global_num_experts=e, renormalize=False, e_score_correction_bias=bias, score_func=score_func)
    torch.cuda.synchronize()
    t4 = time.perf_counter()

    average_fused_moe_time = (t2 - t1) / 100 * 1e3 # ms
    average_iterative_moe_time = (t4 - t3) / 100 * 1e3 # ms
    print(f"batchsize = {m}, score_func = {score_func}, dtype = {dtype}")
    print(f"fused_moe: average time: {average_fused_moe_time:.2f} ms")
    print(f"iterative_moe: average time: {average_iterative_moe_time:.2f} ms")
    print(f"Speedup: {average_iterative_moe_time / average_fused_moe_time:.2f}x")
    

if __name__ == "__main__":
    for _ in range(1):
        benchmark_fused_moe(1, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(2, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(4, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(8, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(16, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(32, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(64, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(128, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(256, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(512, 256, 7168, 256, 8, "softmax", torch.float16)
        benchmark_fused_moe(1024, 256, 7168, 256, 8, "softmax", torch.float16)

        benchmark_fused_moe(1, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(2, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(4, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(8, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(16, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(32, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(64, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(128, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(256, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(512, 256, 7168, 256, 8, "sigmoid", torch.float16)
        benchmark_fused_moe(1024, 256, 7168, 256, 8, "sigmoid", torch.float16)