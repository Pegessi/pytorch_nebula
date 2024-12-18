#! /usr/bin/env bash
set -e

SCRIPT_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_PATH=${SCRIPT_DIR}/build

export ROCM_PATH=/opt/dtk
export PATH=/opt/dtk/bin:/opt/dtk/hip/bin:/opt/dtk/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/lib:/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Only run this if you're compiling for ROCm
# python tools/amd_build/build_amd.py

if [ ! -d "$BUILD_PATH" ]; then
 mkdir "$BUILD_PATH"
fi
rm -rf ${BUILD_PATH}/*

# 编译选项 
# export _GLIBCXX_USE_CXX11_ABI=1
# export USE_NUMPY=1
export USE_ROCM=1
# export USE_LMDB=1
export USE_CUDA=0
# export USE_KINETO=0
# export USE_NINJA=0
export BUILD_TEST=0
# export DEBUG=0
# export MAX_JOBS=64
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# 编译 
python setup.py develop