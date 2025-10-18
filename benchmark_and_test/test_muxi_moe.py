# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest test_muxi_moe.py`.
"""
import time
import pytest
import torch

from muxi_layout_kernels.muxi_moe_planC import fused_moe as planC_moe
from muxi_layout_kernels.utils import fused_moe as iterative_moe


@pytest.mark.parametrize(
    "m",
    [
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        176,
        192,
        208,
        224,
        240,
        256,
        272,
        384,
        512,
        1024,
        2048,
    ],
)  # batchSize
@pytest.mark.parametrize(
    "n,k,e",
    [
        (256, 7168, 256),  # DeepSeek-V3 TP8
        (768, 2048, 128),  # Qwen3-30B-A3B TP1
    ],
)
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("score_func", ["sigmoid"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    score_func: str,
    dtype: torch.dtype,
):
    torch.manual_seed(2025)

    a = torch.rand((m, k), device="cuda", dtype=dtype) / 100
    a1 = a.clone()
    w1 = torch.rand((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.rand((e, k, n), device="cuda", dtype=dtype) / 10
    w1_transposed = w1.clone()
    w2_transposed = w2.clone()
    score = torch.rand((m, e), device="cuda", dtype=dtype)
    # bias = torch.rand((e), device="cuda", dtype=dtype)
    bias = None

    # layoutA
    for i in range(e):
        w1_shaped = w1_transposed[i].view(2 * n // 16, 16, k // 8, 8)
        w1_transposed[i] = (
            w1_shaped.permute(0, 2, 1, 3).contiguous().view(2 * n, k).contiguous()
        )
        w2_shaped = w2_transposed[i].view(k // 16, 16, n // 8, 8)
        w2_transposed[i] = (
            w2_shaped.permute(0, 2, 1, 3).contiguous().view(k, n).contiguous()
        )

    maca_output = planC_moe(
        a,
        w1_transposed,
        w2_transposed,
        score,
        topk,
        renormalize=True,
        use_grouped_topk=True,
        num_expert_group=8,
        topk_group=4,
        gating_bias=bias,
        score_func=score_func,
    )

    iterative_output = iterative_moe(
        a1,
        w1,
        w2,
        score,
        topk,
        8,
        4,
        global_num_experts=e,
        renormalize=True,
        e_score_correction_bias=bias,
        score_func=score_func,
    )

    torch.testing.assert_close(maca_output, iterative_output, atol=0.1, rtol=0.05)
    print("Test passed")


if __name__ == "__main__":
    # torch.set_printoptions(threshold=float('inf'))
    pytest.main()
