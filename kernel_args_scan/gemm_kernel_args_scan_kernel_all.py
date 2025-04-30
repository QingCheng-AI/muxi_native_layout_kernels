import torch
import muxi_layout_kernels
import time
import csv

# Function to test and measure performance for given m, n, k
def test_gemm(m, n, k, warpup_times, benchmark_times, exec_times, kernelParam1, kernelParam2, kernelParam3, kernelId):
    # Initialize tensors
    A = torch.rand(m, k, device='cuda', dtype=torch.float16) * 2 - 1
    B = torch.rand(n, k, device='cuda', dtype=torch.float16) * 2 - 1

    alpha = 1.0
    beta = 0.0

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

    average_gemmex_time = (gemmex_total_execution_time / benchmark_times) * 1e9  # ns

    # Prepare layout A
    A_reshaped = A.view(m // 16, 16, k // 8, 8)
    A_transposed = A_reshaped.permute(0, 2, 1, 3).contiguous()

    B_transposed = muxi_layout_kernels.layoutB(B)

    # Warm-up for gemm_layoutABC
    for _ in range(warpup_times):
        for _ in range(exec_times):
            C = muxi_layout_kernels.gemm_layoutABC_wapper(A_transposed, B_transposed,m, n, k, alpha, beta, kernelParam1, kernelParam2, kernelParam3, kernelId)
        torch.cuda.synchronize()

    # Benchmark gemm_layoutABC
    layoutABC_total_execution_time = 0.0
    layoutABC_total_layoutB_time = 0.0

    B_transposed = muxi_layout_kernels.layoutB(B)

    torch.cuda.synchronize()
    time3 = time.perf_counter()

    for _ in range(benchmark_times):
        for _ in range(exec_times):
            C = muxi_layout_kernels.gemm_layoutABC_wapper(A_transposed, B_transposed,m, n, k, alpha, beta, kernelParam1, kernelParam2, kernelParam3, kernelId)    
    torch.cuda.synchronize()
    time4 = time.perf_counter()
    layoutABC_total_execution_time += (time4 - time3)    

    average_layoutABC_time = (layoutABC_total_execution_time / benchmark_times) * 1e9  # ns

    C_reshaped = C.view(m // 32, n // 16, 4, 16, 8)
    C_transposed = C_reshaped.permute(1, 3, 0, 2, 4).contiguous()
    C_transposed = C_transposed.view(n, m)

    C_cpu = C_transposed.cpu()
    C_expected_cpu = C_expected.cpu()

    # check_passed (simple)
    tolerance = 2.5e-1
    check = True
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
        print(f"Values: gemmEx = {max_diff_value_cpu}, my_kernel = {max_diff_value_expected_cpu}")

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
    # speedup = (average_gemmex_time / average_layoutABC_time)
    layout_mfu = 2.0 * m * n * k / (average_layoutABC_time * 1e-9) / 1e12 / 292
    if max_diff.item() > 1:
        layout_mfu = -2.0

    return m, n, k, average_gemmex_time, average_layoutABC_time, layout_mfu, kernelParam1, kernelParam2, kernelParam3, kernelId, max_diff.item()

def result0(m, n, k, warpup_times, benchmark_times, exec_times, AperWarp, splitN, splitK):

    return m, n, k, -1.0, -1.0, exec_times, APerWarp, splitN, splitK, 0.0


results = []
best_result = []
best_data = []

# m = [4608, 3584, 37888, 3584]
# k = [3584, 3584, 3584, 18944]
# m = [7168]
# k = [256]
m = [3072, 7168, 4608, 7168, 7168, 512]
k = [1536, 2048, 7168, 2304, 256, 7168]
# m = [512, 7168]
# k = [7168, 256]
max_n = 128

warpup_times = 1000  # Reduced warmup for multiple cases
benchmark_times = 1000

scan_count = 0

for exec in range(1, 2):
    for i in range(len(m)):
        for n_ in range(16, max_n + 1, 16):
            kernelId = 1
            for kernelParam1 in range(1, 3, 1):
                for kernelParam2 in range(1, 5, 1):
                    for kernelParam3 in range(1, 5, 1):
                        try:
                            result = m[i], n_, k[i], -1.0, -1.0, exec, kernelParam1, kernelParam2, kernelParam3, kernelId, -1.0
                            result = test_gemm(m[i], n_, k[i], warpup_times, benchmark_times, exec, kernelParam1, kernelParam2, kernelParam3, kernelId)
                        finally:
                            results.append(result)
                        print(f"Tested case m={m[i]}, n={n_}, k={k[i]}")
                        print(result)
                        scan_count += 1
            kernelId = 2
            for kernelParam1 in [64, 128]:
                for kernelParam2 in range(16, 129, 16):
                    for kernelParam3 in [128]:
                        try:
                            result = m[i], n_, k[i], -1.0, -1.0, exec, kernelParam1, kernelParam2, kernelParam3, kernelId, -1.0
                            result = test_gemm(m[i], n_, k[i], warpup_times, benchmark_times, exec, kernelParam1, kernelParam2, kernelParam3, kernelId)
                        finally:
                            results.append(result)
                        print(f"Tested case m={m[i]}, n={n_}, k={k[i]}")
                        print(result)
                        scan_count += 1
            max_mfu = -1.0
            for index in range(scan_count - (32 + 16), scan_count):
                if results[index][5] > max_mfu:
                    max_mfu = results[index][5]
                    best_result_obj = results[index]
            best_result.append([best_result_obj[0], best_result_obj[1], best_result_obj[2], best_result_obj[9], best_result_obj[6], best_result_obj[7], best_result_obj[8]])
            best_data.append(best_result_obj)
            print(f"Best case m={best_result_obj[0]}, n={best_result_obj[1]}, k={best_result_obj[2]}")
            print(best_result_obj)
                

# Save results to CSV
csv_file = "MUXI_GEMM_MFU" + "_scan_result_all_kernel.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["m", "n", "k", "average_gemmex_time","average_layoutABC_time", "mfu_layout", "kernelParam1", "kernelParam2", "kernelParam2", "kernelId", "max_diff_value"])
    writer.writerows(results)

csv_file = "MUXI_GEMM_MFU" + "_scan_best_args_all_kerne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["m", "n", "k", "kernelId", "kernelParam1", "kernelParam2", "kernelParam3"])
    writer.writerows(best_result)

csv_file = "MUXI_GEMM_MFU" + "_scan_best_data_all_kerne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["m", "n", "k", "average_gemmex_time","average_layoutABC_time", "mfu_layout", "kernelParam1", "kernelParam2", "kernelParam2", "kernelId", "max_diff_value"])
    writer.writerows(best_data)

print(f"Results saved to {csv_file}")
