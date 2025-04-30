import torch
import muxi_layout_kernels
import time
import csv

# Function to test and measure performance for given m, n, k
def test_gemm(m, n, k, warpup_times, benchmark_times, exec_times):
    # Initialize tensors
    A = torch.rand(m, k, device='cuda', dtype=torch.float16) * 2 - 1
    B = torch.rand(n, k, device='cuda', dtype=torch.float16) * 2 - 1

    alpha = 1.0
    beta = 0.0

    # Warm-up for gemmEx
    for _ in range(warpup_times):
        for _ in range(exec_times):
            C_expected = muxi_layout_kernels.muxi_hgemm(A, B, alpha, beta)
            torch.cuda.synchronize()

    # Benchmark gemmEx
    gemmex_total_execution_time = 0.0
    for _ in range(benchmark_times):
        time1 = time.perf_counter()
        for _ in range(exec_times):
            C_expected = muxi_layout_kernels.muxi_hgemm(A, B, alpha, beta)
        torch.cuda.synchronize()
        time2 = time.perf_counter()
        gemmex_total_execution_time += (time2 - time1)

    average_gemmex_time = (gemmex_total_execution_time / benchmark_times) * 1e9  # ns

    # Prepare layout A
    A_reshaped = A.view(m // 16, 16, k // 8, 8)
    A_transposed = A_reshaped.permute(0, 2, 1, 3).contiguous()

    B_transposed = muxi_layout_kernels.layoutB(B)

    # Warm-up for gemm_layoutABC
    for _ in range(warpup_times):
        for _ in range(exec_times):
            C = muxi_layout_kernels.muxi_hgemm_layout(A_transposed, B_transposed, alpha, beta)

        torch.cuda.synchronize()

    B_transposed = muxi_layout_kernels.layoutB(B)

    # Benchmark gemm_layoutABC
    layoutABC_total_execution_time = 0.0
    layoutABC_total_layoutB_time = 0.0
    for _ in range(benchmark_times):
        time3 = time.perf_counter()
        
        
        for _ in range(exec_times):
            C = muxi_layout_kernels.muxi_hgemm_layout(A_transposed, B_transposed, alpha, beta)

        torch.cuda.synchronize()
        time4 = time.perf_counter()

        layoutABC_total_execution_time += (time4 - time3)

    average_layoutABC_time = (layoutABC_total_execution_time / benchmark_times) * 1e9  # ns

    C_cpu = C.cpu()
    C_expected_cpu = C_expected.cpu()

    # print(C_expected_cpu)
    # print(C_cpu)

    # check_passed (simple)
    tolerance = 1e-2
    diff = torch.abs(C_cpu - C_expected_cpu)  # 计算绝对差值

    # 找到最大差值及其索引
    max_diff, max_diff_index = diff.flatten().max(dim=0)  # 将张量展平并找到最大值及其索引

    if max_diff.item() > tolerance:  # 检查最大差值是否超过容忍度
        # 获取最大差值对应的C_cpu和C_expected_cpu中的值
        max_diff_value_cpu = C_cpu.flatten()[max_diff_index].item()
        max_diff_value_expected_cpu = C_expected_cpu.flatten()[max_diff_index].item()

        # 输出最大差值的索引和对应的两个值
        print(f"Max difference: {max_diff}")
        print(f"Max difference index: {max_diff_index.item()}")
        print(f"Values: gemmEx = {max_diff_value_expected_cpu}, my_kernel = {max_diff_value_cpu}")
    check = True
    mask = ~torch.isclose(C_cpu, C_expected_cpu, atol=tolerance)
    if mask.any():
        mismatch_index = mask.nonzero(as_tuple=True)
        first_mismatch = (mismatch_index[0][0].item(), mismatch_index[1][0].item())
        C_value = C_cpu[first_mismatch[0], first_mismatch[1]].item()
        C_expected_value = C_expected_cpu[first_mismatch[0], first_mismatch[1]].item()
        print(f"First mismatch index: {first_mismatch}")
        print(f"Values: C = {C_value}, C_expected = {C_expected_value}")
        check = False
    else:
        check = True

    # Calculate optimization
    speedup = (average_gemmex_time / average_layoutABC_time)

    return m, n, k, average_gemmex_time, average_layoutABC_time, speedup, exec_times, max_diff.item()

# Main loop to test multiple cases and save results
results = []
# m = [4608, 3584, 37888, 3584]
# k = [3584, 3584, 3584, 18944]

m = [37888]
k = [3584]
warpup_times = 0  # Reduced warmup for multiple cases
benchmark_times = 1

for exec in range(1, 2):  # Increase execution times for more accurate benchmark
    for i in range(len(m)):
        for n in [16 * 1024, 32 * 1024, 48 * 1024, 64 * 1024, 80 * 1024]:
            result = test_gemm(m[i], n, k[i], warpup_times, benchmark_times, exec)
            results.append(result)
            print(f"Tested case m={m[i]}, n={n}, k={k[i]}")
            print(result)

# Save results to CSV
csv_file = "gemm_benchmark_Layout_muxi_hgemm_results.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["m", "n", "k", "muxi_hgemm_time_ns", "layout_muxi_hgemm_time_ns", "speedup", "exec_times", "max_diff"])
    writer.writerows(results)

print(f"Results saved to {csv_file}")
