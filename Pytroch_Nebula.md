# 环境准备

推荐所有可以指定路径的安装都放在/data目录

## conda

推荐使用Miniconda在本地创建虚拟环境管理

## CUDA cuDNN

目前正常执行的环境是cuda=11.8, cudnn=8.9.2

服务器上的环境是12.2，不确定是否能用，需要本地安装cuda并配置环境变量来正常使用

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
# BUILD_TEST=0 python setup.py bdist_wheel
# use GMLake (暂不可用)
# GMLAKE_ENABLED=1 TORCH_CUDA_ARCH_LIST="8.0" USE_CUDA=1 BUILD_TEST=0 CMAKE_BUILD_TYPE=Debug python setup.py develop
```

debug模式编译时间较久，如果服务器当前有人在使用，建议`export MAX_JOB=64`限制进程数目，避免影响他人使用

之后的操作是建立在源码安装好torch基础上的

## transformer

hugging face的transformer迭代较快，这里使用了23年底一个固定的版本进行开发，主要的依赖在这个里面

```bash
git clone https://github.com/Pegessi/dtr_workspace # 这个仓库其实可以不下载
cd dtr_workspace/hf_train
pip install -r requirements.txt
```

**兼容性修改，这一部分是早期修改的，现在最新的版本其实不一定需要，可以先测试一下不执行这个脚本是否可以执行程序**

同样在`dtr_workspace/hf_train/compatible_change.sh`中，注意先设定CONDA_DIR以及cur_env来替换你本地的环境

主打一个文件修改，逻辑比较简单，但是组织比较复杂，注意新环境只跑一次这个脚本，如果废了建议重开一个环境

或者你可以挨个文件查找对应内容进行修改

**注意这个仓库不跑megatron-LM，直接屏蔽掉文件修改中所有包含megatron的路径**

# 执行测试

单机（主要是deepspeed，请参考这个repo）

https://github.com/Pegessi/Accelerate_LLM

多机（主要是Megatron-LM，请参考这个repo）

https://github.com/Pegessi/Megatron-LM/tree/dtb