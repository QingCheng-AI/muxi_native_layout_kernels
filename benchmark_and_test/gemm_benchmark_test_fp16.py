import torch
import muxi_layout_kernels
import time
import csv

# Function to test and measure performance for given m, n, k
def test_gemm(m, n, k, warpup_times, benchmark_times, exec_times):
    # Initialize tensors
    A = torch.rand(m, k, device='cuda', dtype=torch.float16) * 2 - 1
    B = torch.rand(n, k, device='cuda', dtype=torch.float16) * 2 - 1
    bias = torch.ones(1, m, device='cuda', dtype=torch.float16) * 100

    alpha = 1.0
    beta = 0.0

    # for gemmEx NN
    trans_A = A.t().contiguous()

    # Warm-up for gemmEx
    for _ in range(warpup_times):
        for _ in range(exec_times):
            C_expected = muxi_layout_kernels.gemmEx(trans_A, B, alpha, beta)
    torch.cuda.synchronize()

    # Benchmark gemmEx
    gemmex_total_execution_time = 0.0
    torch.cuda.synchronize()
    time1 = time.perf_counter()
    for _ in range(benchmark_times):
        for _ in range(exec_times):
            C_expected = muxi_layout_kernels.gemmEx(trans_A, B, alpha, beta)
    torch.cuda.synchronize()
    time2 = time.perf_counter()
    gemmex_total_execution_time += (time2 - time1)
    
    C_expected = C_expected + bias

    average_gemmex_time = (gemmex_total_execution_time / benchmark_times) * 1e9  # ns

    # Prepare layout A
    A_reshaped = A.view(m // 16, 16, k // 8, 8)
    A_transposed = A_reshaped.permute(0, 2, 1, 3).contiguous()

    # Warm-up for gemm_layoutABC
    B_transposed = muxi_layout_kernels.layoutB(B)
    for _ in range(warpup_times):
        for _ in range(exec_times):
            # C = muxi_layout_kernels.gemm_layoutABC(A_transposed, B_transposed, alpha, beta, bias)
            C = muxi_layout_kernels.gemm_layoutAB_ContinuousC(A_transposed, B_transposed, alpha, beta, bias)
            # C = muxi_layout_kernels.muxi_hgemm_layout(A_transposed, B_transposed, alpha, beta, bias)
    torch.cuda.synchronize()

    # Benchmark gemm_layoutABC
    layoutABC_total_execution_time = 0.0
    layoutABC_total_layoutB_time = 0.0
    B_transposed = muxi_layout_kernels.layoutB(B)

    torch.cuda.synchronize()
    time3 = time.perf_counter()
    for _ in range(benchmark_times):
        for _ in range(exec_times):
            # C = muxi_layout_kernels.gemm_layoutABC(A_transposed, B_transposed, alpha, beta, bias)
            C = muxi_layout_kernels.gemm_layoutAB_ContinuousC(A_transposed, B_transposed, alpha, beta, bias)
            # C = muxi_layout_kernels.muxi_hgemm_layout(A_transposed, B_transposed, alpha, beta, bias)
    torch.cuda.synchronize()
    time4 = time.perf_counter()        

    layoutABC_total_execution_time += (time4 - time3)
    average_layoutABC_time = (layoutABC_total_execution_time / benchmark_times) * 1e9  # ns

    # C_transposed = muxi_layout_kernels.reLayoutC(C)
    # C_reshaped = C.view(m // 32, n // 16, 4, 16, 8)
    # C_transposed = C_reshaped.permute(1, 3, 0, 2, 4).contiguous()
    # C = C_transposed.view(n, m)

    C_cpu = C.cpu()
    C_expected_cpu = C_expected.cpu()

    # check_passed (simple)
    diff = torch.abs(C_cpu - C_expected_cpu)  # 计算绝对差值

    # 找到最大差值及其索引
    max_diff, max_diff_index = diff.flatten().max(dim=0)  # 将张量展平并找到最大值及其索引

    tolerance = 0.1
    if max_diff.item() > tolerance:  # 检查最大差值是否超过容忍度
        # 获取最大差值对应的C_cpu和C_expected_cpu中的值
        max_diff_value_cpu = C_cpu.flatten()[max_diff_index].item()
        max_diff_value_expected_cpu = C_expected_cpu.flatten()[max_diff_index].item()

        # 输出最大差值的索引和对应的两个值
        print(f"Max difference: {max_diff}")
        print(f"Max difference index: {max_diff_index.item()}")
        print(f"Values: gemmEx = {max_diff_value_expected_cpu}, my_kernel = {max_diff_value_cpu}")


    # Calculate optimization
    speedup = (average_gemmex_time / average_layoutABC_time)
    mfu_layout =  2.0 * m * n * k / (average_layoutABC_time * 1e-9) / 1e12 / 292
    mfu_gemmEx = 2.0 * m * n * k / (average_gemmex_time * 1e-9) / 1e12 / 292

    return m, n, k, mfu_layout, mfu_gemmEx, speedup, exec_times, max_diff.item()

# Main loop to test multiple cases and save results
results = []
# m, k = 32000, 4096  # Fixed m and k
# m = [32000, 12288, 4096, 22016, 4096]
# k = [4096, 4096, 4096, 4096, 11008]  # Fixed m and k

# Qwen2 shapes
m = [4608, 3584, 37888, 3584]
k = [3584, 3584, 3584, 18944]
warpup_times = 1000  # Reduced warmup for multiple cases
benchmark_times = 1000

for exec in range(1, 2):  # Increase execution times for more accurate benchmark
    for i in range(len(m)):
        for n in range(16, 257, 16):
            result = test_gemm(m[i], n, k[i], warpup_times, benchmark_times, exec)
            results.append(result)
            print(f"Tested case m={m[i]}, n={n}, k={k[i]}")
            print(result)

# Save results to CSV
csv_file = "gemm_benchmark_results.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["m", "n", "k", "mfu_layout", "mfu_gemmEx", "speedup", "exec_times", "max_error"])
    writer.writerows(results)

print(f"Results saved to {csv_file}")
