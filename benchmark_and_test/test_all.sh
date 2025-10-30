#!/usr/bin/bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# gcc in the docker images is too new
printf '%s\n' \
  "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" \
  "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" \
  "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" \
  "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" \
  > /etc/apt/sources.list
apt update && apt install -y g++-11

pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install $SCRIPT_DIR/.. --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

python $SCRIPT_DIR/gemm_benchmark_test_bf16.py
python $SCRIPT_DIR/gemm_benchmark_test_fp16.py
python $SCRIPT_DIR/gemm_benchmark_hgemm.py
python $SCRIPT_DIR/gemv_benchmark.py
python $SCRIPT_DIR/gemv_benchmark_soft_fp8.py
pytest $SCRIPT_DIR/test_muxi_moe.py
pytest $SCRIPT_DIR/test_muxi_moe_softfp8.py
python $SCRIPT_DIR/benchmark_muxi_moe.py
