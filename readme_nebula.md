# Install

推荐所有可以指定路径的安装都放在/data目录

## conda

推荐使用Miniconda在本地创建虚拟环境管理

## CUDA cuDNN

目前正常执行的环境是cuda=11.8, cudnn=8.9.2

## torch

torch==2.1.0（源码安装的，之后不要随便pip install torch）

配置torch的详细流程如下，建议文件放在/data目录下，因为编译后的项目较大

```bash
conda create -n [env_name] python=3.10
conda activate [env_name]

git clone https://github.com/Pegessi/pytorch [rename]
cd [rename]
git switch dtb
git submodule sync
git submodule update --init --recursive

conda install cmake ninja
pip install -r requirements.txt
conda install mkl mkl-include
conda install -c pytorch magma-cuda110
make triton
pip install typing_extensions
pip install pyyaml
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# BUILD_TEST=0 CMAKE_BUILD_TYPE=Debug python setup.py develop # debug c++
BUILD_TEST=0 python setup.py develop # release
```

debug模式编译时间较久，如果服务器当前有人在使用，建议`export MAX_JOB=64`限制进程数目，避免影响他人使用

之后的操作是建立在源码安装好torch基础上的

## deepspeed

ref: https://github.com/Pegessi/dtr_workspace/blob/main/hf_train/compatible_change.sh

```sh
# 首先完成前面的pytorch安装
pip install -r requirements.txt # hf_train中的依赖
./compatible_change.sh 

# Megatron-LM accelerate中的安装（2022年的版本）
git clone https://github.com/NVIDIA/apex apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" global-option="--fast_layer_norm" ./
# otherwise
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
pip install git+https://github.com/huggingface/Megatron-LM.git

# torchvision安装，非源码会强制安装很多环境
git clone https://github.com/pytorch/vision torchvision
cd torchvision
git checkout fbb4cc54ed521ba912f50f180dc16a213775bf5c # 0.16.0 for torch2.1.0
conda install -c conda-forge ffmpeg
python setup.py install
```

## Megatron-LM

```sh
# 首先完成前面的pytorch安装 

# apex安装
git clone https://github.com/NVIDIA/apex apex
cd apex
pip install packaging
# change libstdc++.so.6 if necessary [link to the right version of libstdc++.so with GLIBCXX_3.4.30]
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

pip install --user -U nltk

# transformer engine
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git
git submodule update --init --recursive
pip install cmake
pip install .
cd ..

git clone https://github.com/Pegessi/Megatron-LM [rename]
cd Megatron-LM
git checkout dtb
git submodule sync
git submodule update --init --recursive
pip install . # 只安装megatron.core
```

# Test

## Llama2-7B Lora

ref: https://github.com/Pegessi/Accelerate_LLM

## Megatron-LM pretrain

ref: https://github.com/Pegessi/Megatron-LM/tree/dtb