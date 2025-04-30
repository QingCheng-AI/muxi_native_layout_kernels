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
def test_gemm(m, n, k, warmup_times, benchmark_times, blockDimX, KernelId, KernelPara1, KernelParam2, exec_times):
    # Initialize tensors
    # A = torch.rand(m, k, device='cuda', dtype=torch.float16) * 2 - 1
    # B = torch.rand(n, k, device='cuda', dtype=torch.float16) * 2 - 1

    # alpha = 1.0
    # beta = 0.0

    # trans_A = A.t().contiguous()
    # torch.manual_seed(2056973347)

    A = torch.randn(m, k, device='cuda', dtype=torch.bfloat16) / 10
    A = A.to(torch.float8_e4m3fn)
    B = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) / 10
    C = torch.zeros(n, m, device='cuda', dtype=torch.bfloat16)
    # bias = torch.rand(1, m, device='cuda', dtype=torch.bfloat16)
    bias = torch.zeros(1, m, device='cuda', dtype=torch.bfloat16)
    scale_matrix = torch.ones((m + 128 - 1) // 128, (k + 128 - 1) // 128, device='cuda', dtype=torch.float32)

    alpha = 1.0
    beta = 0.0

    scaled_A = scale_tensor(A, scale_matrix)

    trans_A = scaled_A.t().contiguous()

    # Warm-up for gemmEx
    for _ in range(warmup_times):
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

    # Warm-up for gemm_layoutABC
    for _ in range(warmup_times):
        for _ in range(exec_times):
            C = muxi_layout_kernels.gemv_layoutA_wapper(A_transposed, B, m, k, alpha, beta, blockDimX, KernelId, KernelParam1, KernelParam2, scale_matrix)
    torch.cuda.synchronize()


    # Benchmark gemm_layoutABC
    layoutABC_total_execution_time = 0.0
    layoutABC_total_layoutB_time = 0.0

    torch.cuda.synchronize()
    time3 = time.perf_counter()
    for _ in range(benchmark_times):
        for _ in range(exec_times):
            C = muxi_layout_kernels.gemv_layoutA_wapper(A_transposed, B, m, k, alpha, beta, blockDimX, KernelId, KernelParam1, KernelParam2, scale_matrix)

    torch.cuda.synchronize()
    time4 = time.perf_counter()
    layoutABC_total_execution_time += (time4 - time3)    

    average_layoutABC_time = (layoutABC_total_execution_time / benchmark_times) * 1e9  # ns

    C_cpu = C.cpu()
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
    layout_mfu = 2.0 * m * n * k / (average_layoutABC_time * 1e-9) / 1e12 / 292

    gemmEx_mfu = 2.0 * m * n * k / (average_gemmex_time * 1e-9) / 1e12 / 292

    layout_bandwidth = (m * n + n * k + m * k) * 2 / (average_layoutABC_time * 1e-9) / 1e12

    gemmEx_bandwidth = (m * n + n * k + m * k) * 2 / (average_gemmex_time * 1e-9) / 1e12

    bandwidth_usage = (m * n + n * k + m * k) * 2 / (average_gemmex_time * 1e-9) / 1.84 / 1e12

    if(max_diff.item() > 1.0):
        return m, n, k, -2.0, -2.0, KernelId, KernelParam1, KernelParam2, exec, max_diff.item()
    else:
        return m, n, k, average_layoutABC_time, layout_mfu, gemmEx_mfu, blockDimX, KernelId, KernelParam1, KernelParam2, layout_bandwidth, gemmEx_bandwidth, max_diff.item()

    # return m, n, k, average_layoutABC_time, layout_mfu, gemmEx_mfu, APerWarp, splitK, layout_bandwidth, gemmEx_bandwidth, max_diff.item()

def result0(m, n, k, warpup_times, benchmark_times, blockDimX, KernelId, KernelPara1, KernelParam2, exec_times):

    return m, n, k, -1.0, -1.0, -1.0, blockDimX, KernelId, KernelParam1, KernelParam2, -1.0, -1.0, -1.0


results = []
best_result = []

# m = [4608, 3584, 37888, 3584]
# k = [3584, 3584, 3584, 18944]
# m = [3584]
# k = [18944]

#deepseek R1 671B shapes
m = [2112, 3072, 7168, 7168, 512, 4608, 7168]
k = [7168, 1536, 2048, 256, 7168, 7168, 2304]


warmup_times = 1000  # Reduced warmup for multiple cases
benchmark_times = 1000

scan_count = 0

for exec in range(1, 2):  # Increase execution times for more accurate benchmark
    for i in range(len(m)):
        for n_ in range(1, 2, 1):
            for blockDimX in (64, 128, 256, 512, 1024):
                KernelId = 1
                for KernelParam1 in range(1, 9, 1):
                    for KernelParam2 in range(1, 9, 1):
                        result = test_gemm(m[i], n_, k[i], warmup_times, benchmark_times, blockDimX, KernelId, KernelParam1, KernelParam2, exec)
                        results.append(result)
                        print(f"Tested case m={m[i]}, n={n_}, k={k[i]}, KernelId={KernelId}, BLOCKDIMX={blockDimX}")
                        print(result)
                        scan_count += 1
                KernelId = 2
                for KernelParam1 in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
                    KernelParam2 = 0
                    if((KernelParam1 > blockDimX) or (k[i] > 4096)):
                        result = result0(m[i], n_, k[i], warmup_times, benchmark_times, blockDimX, KernelId, KernelParam1, KernelParam2, exec)
                    else:
                        result = test_gemm(m[i], n_, k[i], warmup_times, benchmark_times, blockDimX, KernelId, KernelParam1, KernelParam2, exec)
                    results.append(result)
                    print(f"Tested case m={m[i]}, n={n_}, k={k[i]}, KernelId={KernelId}, BLOCKDIMX={blockDimX}")
                    print(result)
                    scan_count += 1
            max_mfu = -1.0
            for index in range(scan_count - (375), scan_count):
                if results[index][4] > max_mfu:
                    max_mfu = results[index][4]
                    best_result_obj = results[index]
            best_result.append([best_result_obj[0], best_result_obj[2], best_result_obj[6], best_result_obj[7], best_result_obj[8], best_result_obj[9]])
            print(f"Best case m={best_result_obj[0]}, n={best_result_obj[1]}, k={best_result_obj[2]}")
            print(best_result_obj)
            
# Save results to CSV
csv_file = "MUXI_GEMV_MFU" + "_scan_result_soft_fp8.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["m", "n", "k", "average_layoutABC_time", "layout_mfu", "gemmEx_mfu", "blockDimX", "KernelId", 'KernelParam1', "KernelParam2", "layout_bandwidth", "gemmEx_bandwidth", "max_diff.item()"])
    writer.writerows(results)

csv_file = "MUXI_GEMV_MFU" + "_scan_best_args_soft_fp8.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["m", "n", "k", "blockDimX", "KernelId", 'KernelParam1', "KernelParam2"])
    writer.writerows(best_result)

print(f"Results saved to {csv_file}")
results.clear()
best_result.clear()
scan_count = 0