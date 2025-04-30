#!/bin/bash

export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export MACA_LIB_PATH=${MACA_PATH}/mxgpu_llvm/lib
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_LIB_PATH}:$LD_LIBRARY_PATH
export PATH=${MACA_CLANG_PATH}:${MACA_PATH}/bin:$PATH

# source ~/tyt/venv/bin/activate