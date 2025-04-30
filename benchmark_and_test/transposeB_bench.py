import torch
import time

k = 4096
n = 128
B = torch.rand([n, k], device=torch.device('cuda'), dtype=torch.float16)

# warmup
for _ in range(1000):
    B_reshaped = B.view(n // 16, 16, k // 32, 4, 8)
    B_transposed = B_reshaped.permute(2, 0, 3, 1, 4).contiguous()

torch.cuda.synchronize()

# benchmark
total_execution_time = 0.0
for _ in range(1000):
    time1 = time.perf_counter()
    
    B_reshaped = B.view(n // 16, 16, k // 32, 4, 8)
    B_transposed = B_reshaped.permute(2, 0, 3, 1, 4).contiguous()

    torch.cuda.synchronize()
    
    time2 = time.perf_counter()
    
    execution_time = (time2 - time1)
    total_execution_time += execution_time

# 计算平均时间
average_execution_time = (total_execution_time / 1000) * 1e9
print(f"Average execution time: {average_execution_time:.2f} ns")