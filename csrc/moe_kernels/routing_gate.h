#pragma once

#include "fused_topK.h"
#include "group_gemm_utils.h"
// #include "../routing_gate.h"

// TODO: softmax for expert weight compute and top-k

// TODO: make metadata with expert_id and expert weight for group gemm
// (expertsIds)

// TODO: implement expert select
// score_fun: 0 for softmax, 1 for sigmoid
template <typename Taccum, typename A>
void routing_gate(A *gating_output, int score_fun, int num_experts,
                  int batch_size, int topK, int n_groups, int topK_groups,
                  int hidden_size, int *expertsIds, A *selected_experts_weights,
                  A *load_balance_bias) {
    A *softmax_experts_score;
    mcMalloc((void **)&softmax_experts_score,
             sizeof(A) * batch_size * num_experts);

    A *group_scores;
    mcMalloc((void **)&group_scores, sizeof(A) * batch_size * n_groups);

    A *dev_selected_experts_group_weights;
    mcMalloc((void **)&dev_selected_experts_group_weights,
             sizeof(A) * batch_size * topK_groups);

    int *dev_selected_experts_group_index;
    mcMalloc((void **)&dev_selected_experts_group_index,
             sizeof(int) * batch_size * topK_groups);

    A *dev_selected_experts_weights;
    mcMalloc((void **)&dev_selected_experts_weights,
             sizeof(A) * batch_size * topK);

    int *dev_expertsIds;
    mcMalloc((void **)&dev_expertsIds, sizeof(int) * batch_size * topK);

    // just for deepseek_V3 and deepseek_R1
    if (n_groups == 8 && topK_groups == 4 && topK == 8 && num_experts == 256 &&
        (score_fun == 0 || score_fun == 1)) {
        fused_softmax_topk::fused_softmax_topk_launcher<A>(
            gating_output, score_fun, batch_size, n_groups, topK_groups,
            dev_expertsIds, dev_selected_experts_weights, topK, num_experts,
            load_balance_bias);
    } else {
        assert(
            false &&
            "now just support deepseek_V3 and deepseek_R1 (for num_experts == "
            "256 and n_groups == 8 and topK_groups == 4 and topK == 8 and "
            "score_fun == 'softmax' or 'sigmoid')");
    }

    mcMemcpy(selected_experts_weights, dev_selected_experts_weights,
             sizeof(A) * batch_size * topK, mcMemcpyDeviceToHost);
    mcMemcpy(expertsIds, dev_expertsIds, sizeof(int) * batch_size * topK,
             mcMemcpyDeviceToHost);

    mcFree(softmax_experts_score);
    mcFree(dev_selected_experts_weights);
    mcFree(dev_expertsIds);
    mcFree(group_scores);
    mcFree(dev_selected_experts_group_weights);
    mcFree(dev_selected_experts_group_index);
}