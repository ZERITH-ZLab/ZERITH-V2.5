# 仿真公共配置

[项目主页](../README_zh-CN.md) · [English](simulation.md)

本文仅记录三个仿真基准共享的环境、编码器资产、训练参数和缓存约定。
先选择下面的独立流程；每份均包含发布权重评测和自行训练后的评测。

| 基准 | 独立流程 |
| :--- | :--- |
| LIBERO | [标准 40 任务：权重、数据、评测与训练](simulation/libero_zh-CN.md) |
| LIBERO-Plus | [鲁棒性基准：权重、资产、评测与训练](simulation/libero-plus_zh-CN.md) |
| MetaWorld | [MT50：权重、数据、评测与训练](simulation/metaworld_zh-CN.md) |

下文所有命令从源码仓库根目录执行。`conda run --cwd simulation` 用于在正确的
工作目录及环境运行命令；新终端仍需先进入源码仓库根目录。

## 环境

模型服务使用独立的 GPU 环境。推理示例采用 CUDA 12.8，训练参考环境采用 CUDA 13.0，驱动须与所选构建兼容；
仅评测发布权重时无需 Megatron、训练演示或视觉缓存。

```bash
conda create --override-channels -c conda-forge -n zerith-inference python=3.12 pip -y
conda activate zerith-inference
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r simulation/requirements-inference.txt
```

自行训练时另外创建训练环境：

```bash
conda create --override-channels -c conda-forge -n zerith-train python=3.12 pip -y
conda run --no-capture-output -n zerith-train python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
conda run --no-capture-output -n zerith-train python -m pip install megatron-core==0.19.0
conda run --no-capture-output -n zerith-train python -m pip install -r simulation/requirements.txt
```

模拟器客户端使用各基准文档指定的独立环境，避免其旧版依赖与模型环境冲突。
LIBERO 与 LIBERO-Plus 也分别安装，配置各自的 `LIBERO_CONFIG_PATH`。

## 所需资产

以下编码器资产由三个基准共享，仅需下载一次；数据准备与推理须使用相同版本。
在 `zerith-inference` 环境中执行：

```bash
mkdir -p simulation/assets
git clone https://github.com/Wan-Video/Wan2.2.git simulation/assets/Wan2.2
git -C simulation/assets/Wan2.2 checkout 42bf4cfaa384bc21833865abc2f9e6c0e67233dc
git clone https://github.com/facebookresearch/vggt.git simulation/assets/vggt
git -C simulation/assets/vggt checkout a288dd0f14786c93483e45524328726ab7b1b4ce
hf download Wan-AI/Wan2.2-TI2V-5B \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir simulation/assets/Wan2.2-TI2V-5B
hf download facebook/VGGT-1B \
  --revision 860abec7937da0a4c03c41d3c269c366e82abdf9 --local-dir simulation/assets/VGGT-1B
```

各基准文档集中列出本基准的权重、仿真资产及可选训练数据下载命令。
发布权重可直接评测，推理采用 FP32。

## 缓存与动作约定

数据准备生成训练所需的归一化统计和视觉、语言缓存。训练与推理应使用同一版本的编码器。
中断后在原命令上加 `--resume`；更换数据、编码器或配置后使用新的输出目录。
同一个输出目录只运行一个准备进程，且与源数据、编码器目录分开。

LIBERO/Plus 使用原生 OSC7，MetaWorld 使用裁剪后的 Native4。
模型预测 16 步，评测执行前 4 步。

## 训练参数

`--gpus` 是每节点 GPU 数，`--batch-size` 是每卡样本数。
**全局 batch = 节点数 × 每节点卡数 × 每卡 batch**。Launcher 使用 `torchrun`
与 Megatron-FSDP，不使用梯度累积。改变卡数会改变全局 batch，学习率不会
自动缩放；应根据显存选择 batch。

生成的 recipe 默认参数如下：

| 设置 | 默认值 |
| --- | --- |
| LIBERO / LIBERO-Plus | 30,000 steps |
| MetaWorld | 50,000 steps |
| 基础学习率 | `1e-5` |
| Wan / Reader backbone LR 倍率 | 1 |
| Reader heads / geometry / G4D LR 倍率 | 10 |
| AdamW | betas `(0.9, 0.95)`，weight decay `0.01` |
| 调度 | cosine，5% warmup，最低 LR 为初始值的 10% |
| 梯度裁剪 / 精度 | 1.0 / BF16 |
| 动作损失到 Wan 的梯度 | 前 5% 步数为 0，随后 5% 线性增加到 1 |
| Checkpoint | 每 1,000 步，保留两个恢复点 |

可通过 `--steps`、`--learning-rate`、`--batch-size` 和 `--workers` 修改参数。
加 `--dry-run` 只打印生成的配置和全局 batch，不启动训练。

### 多机与恢复

每个节点运行所选基准文档中的训练命令，加上 `--nodes N --node-rank R --master-addr TRAINING_HOST --master-port 29500`。节点编号为 `0…N-1`，各节点的
数据、权重路径须一致，输出目录须共享。`--gpus` 是每节点卡数；四节点、每节点
八卡、每卡 batch 10 对应全局 batch 320。

恢复时保留拓扑、batch、数据和超参数，在原命令加 `--resume`。
最新完整 checkpoint 恢复优化器、调度器、随机状态和 sampler 游标。
只从自己创建或信任来源的训练目录恢复。

### 初始化与导出

默认使用发布权重初始化，也可用 `--checkpoint` 指定其他架构兼容的权重。
各基准文档给出数据准备、训练、导出和服务命令。

## 评测产物

输出目录包含 `episodes.jsonl`；完整结束后还会生成 `summary.json` 并打印
`EVAL_SUMMARY`，包含完成次数、成功次数和成功率（0–1）。
每个服务进程同时只处理一个 episode。

## 可选 FlashAttention-3

默认可使用 PyTorch SDPA。Hopper GPU 可选编译[自定义扩展](../simulation/fa2_custom_mask/)；
需要兼容的 CUDA toolkit（含 `nvcc`）和 C++ 编译器：

```bash
conda run --no-capture-output -n zerith-train python -m pip install ninja
conda run --no-capture-output -n zerith-train --cwd simulation python -c 'from fa2_custom_mask import prebuild_flash_attention3; print(prebuild_flash_attention3())'
```

编译失败会回退 SDPA；设置 `ZERITH_REQUIRE_FLASH_ATTENTION3=1` 可要求使用该扩展。

## 开发检查

```bash
conda run --no-capture-output -n zerith-train --cwd simulation python -m pytest -q tests
```

单元测试覆盖 codec、掩码、采样、模型加载和导出物回读。
完整训练与闭环评测需要实际模型资产和模拟器环境，单元测试不代表 benchmark 效果。
