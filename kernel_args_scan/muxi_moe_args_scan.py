# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest test_muxi_moe.py`.
"""
import time
import pytest
import torch 
import csv

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
    score: torch.tensor,
    AperWarp: int = 2,
    splitK: int = 1,
    # tile_m_1: int = 128,
    # tile_n_1: int = 16,
    # tile_k_1: int = 128,
    tile_m_2: int = 128,
    tile_n_2: int = 16,
    tile_k_2: int = 128,
):
    a = torch.rand((m, k), device="cuda", dtype=dtype) / 100
    a1 = a.clone()
    w1 = torch.rand((e, 2 * n, k), device="cuda", dtype=dtype) / 100
    # w1 = torch.rand((e, k, 2 * n), device="cuda", dtype=dtype) / 100
    w2 = torch.rand((e, k, n), device="cuda", dtype=dtype) / 100
    w1_transposed = w1.clone()
    w2_transposed = w2.clone()
    # bias = torch.rand((e), device="cuda", dtype=dtype)
    bias = None

    # layout weights for the MACA operation
    for i in range(e):
        w1_shaped = w1_transposed[i].view(2 * n // 16, 16, k // 8, 8)
        w1_transposed[i] = w1_shaped.permute(0, 2, 1, 3).contiguous().view(2 * n, k).contiguous()
        w2_shaped = w2_transposed[i].view(k // 16, 16, n // 8, 8)
        w2_transposed[i] = w2_shaped.permute(0, 2, 1, 3).contiguous().view(k, n).contiguous()


    for _ in range(100):
        maca_output = planC_moe(a, w1_transposed, w2_transposed, score, topk, renormalize=False, use_grouped_topk=True, num_expert_group=8, topk_group=4, gating_bias=bias, score_func=score_func,
                                AperWarp=AperWarp, splitK=splitK,
                                tile_m_2=tile_m_2, tile_n_2=tile_n_2, tile_k_2=tile_k_2)
    torch.cuda.synchronize()

    t1 = time.perf_counter()
    for _ in range(100):
        maca_output = planC_moe(a, w1_transposed, w2_transposed, score, topk, renormalize=False, use_grouped_topk=True, num_expert_group=8, topk_group=4, gating_bias=bias, score_func=score_func,
                                AperWarp=AperWarp, splitK=splitK,
                                tile_m_2=tile_m_2, tile_n_2=tile_n_2, tile_k_2=tile_k_2)
    
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    # for _ in range(100):
    #     iterative_output = iterative_moe(a1, w1, w2, score, topk, 8, 4, global_num_experts=e, renormalize=False, e_score_correction_bias=bias, score_func=score_func)
    # torch.cuda.synchronize()

    # t3 = time.perf_counter()
    # for _ in range(100):
    #     iterative_output = iterative_moe(a1, w1, w2, score, topk, 8, 4, global_num_experts=e, renormalize=False, e_score_correction_bias=bias, score_func=score_func)
    # torch.cuda.synchronize()
    # t4 = time.perf_counter()
    
    # print(torch.max(torch.abs(maca_output - iterative_output)))
    average_fused_moe_time = (t2 - t1) / 100 * 1e3 # ms
    # average_iterative_moe_time = (t4 - t3) / 100 * 1e3 # ms
    print(f"batchsize = {m}, score_func = {score_func}, dtype = {dtype}")
    print(f"fused_moe: average time: {average_fused_moe_time:.2f} ms")
    # print(f"iterative_moe: average time: {average_iterative_moe_time:.2f} ms")
    # print(f"Speedup: {average_iterative_moe_time / average_fused_moe_time:.2f}x")
    return m, n, k, average_fused_moe_time, AperWarp, splitK, tile_m_2, tile_n_2, tile_k_2

if __name__ == "__main__":
    n = [768]
    k = [2048]
    max_m = 256
    min_time = 1000
    best_res = None
    best_data = []
    expert_num = 64 
    topk_num = 8
    for m_ in range(16, max_m + 1, 16):
        for (n_, k_) in zip(n, k):
            score = torch.rand((m_, expert_num), device="cuda", dtype=torch.bfloat16)
            for AperWarp in [1, 2, 4]:
                for splitK in [1, 2]:
                    for tile_m_2 in [64, 128, 256]:
                        for tile_n_2 in [16]:
                            for tile_k_2 in [128]:
                                res = benchmark_fused_moe(m_, n_, k_, expert_num, topk_num, "softmax", torch.bfloat16, score,
                                                            AperWarp=AperWarp, splitK=splitK,
                                                            tile_m_2=tile_m_2, tile_n_2=tile_n_2, tile_k_2=tile_k_2)
                                if res[3] < min_time:
                                    min_time = res[3]
                                    best_res = res
            min_time = 1000
            best_data.append(best_res)


    csv_file = "MUXI_MOE_TIME" + "_scan_best_data_all_kernel.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["m", "n", "k", "average_fused_moe_time", "kernelParam1", "kernelParam2", "kernelParam3", "kernelParam4", "kernelParam5", "kernelParam6"])
        writer.writerows(best_data)

    print(f"Results saved to {csv_file}")  


    # score = torch.rand((64, expert_num), device="cuda", dtype=torch.bfloat16)
    # res = benchmark_fused_moe(64, 768, 2048, expert_num, topk_num, "softmax", torch.bfloat16, score,
    #                             1, 1, 128, 16, 128)                        