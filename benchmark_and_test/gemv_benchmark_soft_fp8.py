import torch
import muxi_layout_kernels
import time
import csv

def scale_tensor(weight, scale):
    m, k = weight.shape
    result = weight.clone().to(torch.bfloat16)
    for i in range(0, m, 128):
        for j in range(0, k, 128):
            # 找到 scale 矩阵中对应的元素
            scale_i = i // 128
            scale_j = j // 128
            scale_factor = scale[scale_i, scale_j]
            # 对 weight 中 (128, 128) 方块进行缩放
            result[i:i + 128, j:j + 128] *= scale_factor
    return result


# Function to test and measure performance for given m, n, k
def test_gemm(m, n, k, warpup_times, benchmark_times, exec_times):
    # Initialize tensors
    torch.manual_seed(2025)
    A = torch.randn(m, k, device='cuda', dtype=torch.bfloat16) / 10
    A = A.to(torch.float8_e4m3fn)
    B = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) / 10
    C = torch.zeros(n, m, device='cuda', dtype=torch.bfloat16)
    bias = torch.rand(1, m, device='cuda', dtype=torch.bfloat16)
    # bias = torch.zeros(1, m, device='cuda', dtype=torch.bfloat16)
    scale_matrix = torch.randn((m + 128 - 1) // 128, (k + 128 - 1) // 128, device='cuda', dtype=torch.float32)

    alpha = 1.0
    beta = 0.0

    scaled_A = scale_tensor(A, scale_matrix)

    trans_A = scaled_A.t().contiguous()


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
    for _ in range(warpup_times):
        for _ in range(exec_times):
            C = muxi_layout_kernels.gemv_layoutA(A_transposed, B, alpha, beta, scale_matrix, bias)
    torch.cuda.synchronize()


    # Benchmark gemm_layoutABC
    layoutABC_total_execution_time = 0.0
    layoutABC_total_layoutB_time = 0.0

    torch.cuda.synchronize()
    time3 = time.perf_counter()
    for _ in range(benchmark_times):
        for _ in range(exec_times):
            C = muxi_layout_kernels.gemv_layoutA(A_transposed, B, alpha, beta, scale_matrix, bias)

    torch.cuda.synchronize()
    time4 = time.perf_counter()
    layoutABC_total_execution_time += (time4 - time3)    

    average_layoutABC_time = (layoutABC_total_execution_time / benchmark_times) * 1e9  # ns

    C_cpu = C.cpu()
    C_expected_cpu = C_expected.cpu()

    # check_passed (simple)
    tolerance = 1e-1
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
    layout_mfu = 2.0 * m * n * k / (average_layoutABC_time * 1e-9) / 1e12 / 292

    bandwidth_usage = (m * n + n * k + m * k) * 2 / (average_layoutABC_time * 1e-9) / 1.84 / 1e12

    return m, n, k, average_layoutABC_time, layout_mfu, bandwidth_usage, exec_times, check, max_diff.item()

def result0(m, n, k, warpup_times, benchmark_times, exec_times, AperWarp, splitN, splitK):

    return m, n, k, -1.0, -1.0, exec_times, APerWarp, splitN, splitK, False, 0.0


results = []
best_result = []
# # Llama2_7B
# # Main loop to test multiple cases and save results
# m = [32000, 12288, 4096, 22016, 4096]
# k = [4096, 4096, 4096, 4096, 11008]  # Fixed m and k

#Qwen2_7B shapes
# m = [4608, 3584, 37888, 3584]
# k = [3584, 3584, 3584, 18944]

#deepseek R1 671B shapes
m = [2112, 3072, 7168, 7168, 512, 4608, 7168]
k = [7168, 1536, 2048, 256, 7168, 7168, 2304]

warpup_times = 1000  # Reduced warmup for multiple cases
benchmark_times = 1000

scan_count = 0

for exec in range(1, 2):  # Increase execution times for more accurate benchmark
    for i in range(len(m)):
        for n_ in range(1, 2, 1):
            try:
                result = m[i], n_, k[i], -1.0, -1.0, exec, False
                result = test_gemm(m[i], n_, k[i], warpup_times, benchmark_times, exec)
            finally:
                results.append(result)
            print(f"Tested case m={m[i]}, n={n_}, k={k[i]}")
            print(result)
                

# Save results to CSV
# csv_file = "layout_GEMV_MFU_" + "Baichuan_MOE" + "_scan_more.csv"
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["m", "n", "k","average_layoutABC_time", "mfu_layout", "bandwidth_usage", "exec_times", "APerWarp", "SplitN", "SplitK", "check_passed", "max_diff_value"])
#     writer.writerows(results)

# print(f"Results saved to {csv_file}")
