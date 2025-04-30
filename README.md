# muxi_native_layout_kernels

本项目包含一系列面向沐曦（MetaX）GPU，利用 native layout 手段优化的 GeMV、窄 GeMM、fused MoE 算子。

## Native layout 优化简介

传统的矩阵乘类算子需要在其内部进行多次 layout 转换，包括将输入转换成 mma 需要的 layout、调用 mma 做 tensor core 计算、把 mma 的 layout 转成输出。而其中 mma 的 layout 与具体硬件以及不同数据类型和大小的 mma 指令相关。下图是一个直观例子：

![](/docs/mma_layout.png)

如图是一个 16x16x16 大小的 mma，要求 A 矩阵按左侧子图排列，B 和 C 矩阵按右侧子图排列。左侧子图表示矩阵第 0 行第 0 至 3 列要放在线程 T0，4 至 7 列要放在线程 T16，以此类推。

上述转换需保证在 global memory 上连续、在 shared memory 上没有 bank conflict 等要求，增加了算子的运行时开销，也增加了算子开发的复杂性。本项目通过引入适当的预处理步骤，让 tensor 在 global memory 中直接满足 mma 的要求，简化此类计算过程。

## 本项目功能

- 基于 native layout 设计的 GeMV、窄 GeMM、fused MoE 算子。
- 支持 float16、bfloat16 输入，及软 float8 在线转换（即 weight 采用 float8 输入，activations 采用 bfloat16 输入，计算时按位转为bfloat16 计算）。

- 支持对算子中的超参数或 kernel 变种选取进行离线调优，记录于 csv 文件中。调优脚本见 `kernel_args_scan/` 目录。


## 安装运行
安装：

```sh
pip install .
```

简单测试：

```sh
python3 benchmark_and_test/gemm_benchmark.py
python3 benchmark_and_test/gemv_benchmark.py
python3 benchmark_and_test/benchmark_muxi_moe.py
```

## Layout 说明

窄 GeMM kernel 有以下三种，可参考 `benchmark_and_test/gemm_benchmark.py`:
* `gemm_layoutABC`： 输入 A、B 均为 native layout 形式，输出 C 也为 native layout 形式。
* `gemm_layoutAB_ContinuousC`：输入 A、B 均为 native layout 形式，输出 C 为列主序。
* `gemm_layoutA`：输入 A 为 native layout形式，输入 B 为列主序，输出C为列主序。

GeMV kernel 的输入 A 矩阵为 native layout 形式，输入向量 x 为连续形式（非 native layout），输出 y 为连续形式，可参考`benchmark_and_test/gemv_benchmark.py`。

Fused MoE 包括专家选择、专家计算两部分，其中专家权重矩阵输入为 native layout 形式输入，activations 输入为列主序输入，输出为列主序输出。可参考 `benchmark_and_test/test_muxi_moe.py`。

## 许可证

本项目采用 Apache License v2.0 许可证 - 详见 [LICENSE](/LICENSE) 文件。

## 致谢

特此感谢沐曦（MetaX）对本项目开发的支持！
