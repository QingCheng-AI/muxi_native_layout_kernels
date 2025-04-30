# kernel_args_scan_scripts

gemm_kernel_args_scan.py : GEMM kernel参数扫描脚本，在m,k列表处定义需要扫描的shape（如下，m、k一一对应），n_max处定义最大n值，将从n = 16起扫描到n = max_n，步长值为16

m = [4608, 3584, 37888, 3584]
k = [3584, 3584, 3584, 18944]
max_n = 16

输出为两个csv文件，MUXI_GEMM_MFU_scan_result.csv为所有扫描的shape即kernel对应的性能总表，MUXI_GEMM_MFU_scan_best_args.csv为每个shape对应的最佳性能算子参数，可以将其内容直接完整复制到extension中原GEMM参数选择表的末尾


gemv_kernel_args_scan.py : 同上

## Getting started

grun python gemm_kernel_args_scan.py
grun python gemv_kernel_args_scan.py
