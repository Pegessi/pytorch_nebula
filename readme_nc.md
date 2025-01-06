> [!NOTE]
>
> 本文件是[原教程](https://github.com/Pegessi/pytorch/blob/dtb/readme_nebula.md)的摘录，部分地方略有修改并增加了画图相关的使用方法
>
> TODO: Megatron-LM的安装BUG记录

# Install

推荐所有可以指定路径的安装都放在/data目录

## conda

推荐使用Miniconda在本地创建虚拟环境管理

## CUDA cuDNN

目前正常执行的环境是cuda=11.8, cudnn=8.9.2

> 如果遇到“error: 'CUSPARSE COMPUTE TF32’ was not declared in this scope”，可能要下载libcusparse然后放在cuda对应的目录里

## torch

torch==2.1.0（源码安装的，之后不要随便pip install torch）

配置torch的详细流程如下，建议文件放在/data目录下，因为编译后的项目较大

```bash
conda create -n [env_name] python=3.10
conda activate [env_name]

git clone https://github.com/Pegessi/pytorch [rename]
cd [rename]
git switch dtb_nc
git submodule sync
git submodule update --init --recursive

conda install cmake ninja
pip install -r requirements.txt # numpy<2
conda install mkl mkl-include
conda install -c pytorch magma-cuda110
make triton
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# BUILD_TEST=0 CMAKE_BUILD_TYPE=Debug python setup.py develop # debug c++
BUILD_TEST=0 python setup.py develop # release
```

debug模式编译时间较久，如果服务器当前有人在使用，建议`export MAX_JOB=64`限制进程数目，避免影响他人使用

之后的操作是建立在源码安装好torch基础上的

### 画图

画图功能目前适配了deepspeed中的llama（单卡）

首先，用debug模式重新编译pytorch：

```bash
cd pytorch
rm -rf build
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
BUILD_TEST=0 CMAKE_BUILD_TYPE=Debug python setup.py develop # debug c++
```

安装画图需要的库：

```bash
pip install python-igraph matplotlib
conda install -c conda-forge pycairo
```

然后，完成下面[deepspeed的运行环境配置](##deepspeed)，并下载对应的repo：`git clone https://github.com/Pegessi/Accelerate_LLM.git`

，并对默认的脚本`.../Accelerate_LLM_old/Llama_sft/run_llama.sh`做一些修改：

```bash
export DCR_LOCK_ENABLE=1  # 打开DCR
export COUNTER_RENUMBER=1 # 添加一个新的环境变量，表示启用重新编号
# 运行参数调小
python sft_Llama2.py  \
--model_name=$MODEL_PATH \
--dataset_name=$DATASET_PATH \
--training_args.per_device_train_batch_size=4 \
--training_args.per_device_eval_batch_size=1  \
--training_args.gradient_accumulation_steps=2 \
--training_args.mem_budget=8    \
--training_args.max_steps=1     \
--training_args.warmup_steps=1
```

接着尝试训练：

```bash
./run_llama.sh deepspeed_single_node.yaml 1
```

然后复制刚生成的LOG文件路径（例如：/home/yourname/Accelerate_LLM/Llama_sft/2024-12-16-20-34-11-1428010），在pytorch文件夹中找到`plot_nc.py`，修改python脚本的一些画图参数之后，运行：

```bash
python ./plot_nc.py [LOG文件路径]
# 例如：python ./plot_nc.py /home/yourname/Accelerate_LLM/Llama_sft/2024-12-16-20-34-11-1428010
```

## deepspeed

- ref: https://github.com/Pegessi/dtr_workspace/blob/main/hf_train/

- ref: https://github.com/Pegessi/dtr_workspace/blob/main/hf_train/compatible_change.sh

- > 在安装apex的时候：如果遇到：“ImportError: .../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/wanghuibing/pytorch/torch/lib/libtorch_python.so)”，原因是libstdc++.so.6 版本太低，缺少 GLIBCXX_3.4.30，导致无法加载动态库。可以通过Conda升级 libstdc++ 到更高版本，满足 GLIBCXX_3.4.30 的需求：
  > conda install -c conda-forge libstdcxx-ng

```sh
# 首先完成前面的pytorch安装
pip install -r requirements.txt # hf_train中的依赖
./compatible_change.sh # 兼容DTR脚本

# Megatron-LM + accelerate的安装（2022年的集成版本）
git clone https://github.com/NVIDIA/apex apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

pip install git+https://github.com/huggingface/Megatron-LM.git

# torchvision安装，非源码会强制安装很多环境
git clone https://github.com/pytorch/vision torchvision
cd torchvision
git checkout fbb4cc54ed521ba912f50f180dc16a213775bf5c # 0.16.0 for torch2.1.0
conda install -c conda-forge "ffmpeg<6" # 默认下载ffmpeg=7.x，顺带更新mkl=2025.x安装不了这个torchvision
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