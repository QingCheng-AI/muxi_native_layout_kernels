#pragma once

#include "fused_topK.h"
#include "group_gemm_utils.h"

// score_fun: 0 for softmax, 1 for sigmoid
template <typename Taccum, typename A>
void fused_routing_gate(A *gating_output, int score_fun, int num_experts,
                        int batch_size, int topK, int n_groups, int topK_groups,
                        int hidden_size, int *expertsIds,
                        A *selected_experts_weights, A *load_balance_bias) {
    // just for deepseek_V3 and deepseek_R1
    if (n_groups == 8 && topK_groups == 4 && topK == 8 && num_experts == 256 &&
        (score_fun == 0 || score_fun == 1)) {
        fused_softmax_topk::fused_softmax_topk_launcher<A>(
            gating_output, score_fun, batch_size, n_groups, topK_groups,
            expertsIds, selected_experts_weights, topK, num_experts,
            load_balance_bias);
    } else {
        assert(
            false &&
            "now just support deepseek_V3 and deepseek_R1 (for num_experts == "
            "256 and n_groups == 8 and topK_groups == 4 and topK == 8 and "
            "score_fun == 'softmax' or 'sigmoid')");
    }
}