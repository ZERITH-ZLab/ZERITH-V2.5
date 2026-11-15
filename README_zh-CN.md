<p align="center"><img src="media/zerith-logo.png" alt="Zerith Robotics 标志" width="150"></p>
<h1 align="center">ZERITH 2.5</h1>
<p align="center"><strong>From Predictive Power to Policy Learnability</strong></p>
<p align="center">面向机器人控制的视频–动作模型</p>
<p align="center">
  <a href="https://zerith-25-review.kindpetal17.chatgpt.site/">📖 研究博客</a> &nbsp;·&nbsp;
  <a href="https://zerith-25-review.kindpetal17.chatgpt.site/assets/ZERITH-2.5-v1.0.pdf">📄 技术报告</a> &nbsp;·&nbsp;
  <a href="#模型权重">🤗 模型权重</a> &nbsp;·&nbsp;
  <a href="#快速开始">🚀 快速开始</a>
</p>
<p align="center">
  <a href="README.md">English</a> &nbsp;/&nbsp; <strong>简体中文</strong>
</p>

<div align="center">

https://github.com/user-attachments/assets/1affaaa2-3db4-4007-8057-791f7171996b

</div>

## 模型概览

**ZERITH 2.5** 是一个面向机器人操作的视频–动作模型，约有 **6.774B 可训练参数**
（不含冻结的视觉、语言与几何编码器）。模型利用多视角观测、语言指令和机器人状态，
借助视频生成模型的预测特征生成动作。

我们关注的核心问题是：**视频生成先验在什么条件下能帮助机器人学会行动？**
为此，我们固定示范数据和总训练更新次数，研究预测训练如何影响后续动作学习。
受控实验得到以下结论：

- **目标域预测适配能改善策略的可学习性。** 在联合学习视频与动作之前，先对视频模型进行预测适配，在三种几何条件设置下，LIBERO-Plus 成功率提高 **17.55–19.16 个百分点**；每种设置均使用三个训练种子评测。
- **预测监督的方式很关键。** 覆盖较宽噪声范围的适配带来明显收益；仅在部署使用的纯噪声水平上适配，无法复现这一收益。
- **收益跟随适配后的视频模型。** 适配权重保留在视频模型中时，收益得以保留；仅用它们初始化动作策略时，收益消失。

研究方法、受控实验和进一步分析详见[研究博客](https://zerith-25-review.kindpetal17.chatgpt.site/)
和[技术报告](https://zerith-25-review.kindpetal17.chatgpt.site/reader.html)。

## 最新动态

- **2026-11-04：** [ZERITH 2.5 研究预览](https://zerith-25-review.kindpetal17.chatgpt.site/)介绍了模型、基准结果和真机演示。

## 基准测试结果

### 仿真基准

| 基准 | ZERITH 2.5 | π₀ | π₀.₅ | OpenVLA-OFT | Fast-WAM | Cosmos Policy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [LIBERO](docs/simulation/libero_zh-CN.md) | **98.95%** | 94.20% | 96.90% | 97.10% | 97.60% | 98.50% |
| [LIBERO-Plus](docs/simulation/libero-plus_zh-CN.md) | **93.70%** | 53.60% | 85.70% | 69.60% | 49.14% | 82.20% |
| [MetaWorld MT50](docs/simulation/metaworld_zh-CN.md) | **92.3%** | 47.9%† | 81.6%† | — | — | — |

表中数值均为任务成功率，ZERITH 2.5 为报告中的**完整系统**结果；基线保留各自
来源的评测协议。† MetaWorld 基线对四个难度组的成功率等权平均，ZERITH 2.5
则按评测 episode 汇总成功率。模型变体、任务范围及比较条件见
[报告表 4–5](https://zerith-25-review.kindpetal17.chatgpt.site/reader.html#page=12)。
“—”表示未报告结果。点击基准名称可查看评测指南。

### 真机评测与演示

| H1 任务 | 每个模型的试验次数 | ZERITH 2.5 | π₀.₅ | Fast-WAM |
| :--- | :---: | :---: | :---: | :---: |
| 抓放 | 100 | **97%** | 95% | 84% |
| 套袋 | 20 | **40%** | 35% | 15% |
| 双臂装配 | 20 | **10%** | **10%** | 0% |

成功率来自[报告表 6](https://zerith-25-review.kindpetal17.chatgpt.site/reader.html#page=14)。
[演示视频](https://zerith-25-review.kindpetal17.chatgpt.site/#real-world)展示了套袋、
SKU 分拣和双臂装配，各片段标有播放速度；SKU 分拣为定性展示。

## 模型权重

以下 **ZERITH 2.5** 权重覆盖三个仿真基准及 H1 平台。

| 模型 | 用途说明 | 权重下载 | 使用指南 |
| :--- | :--- | :---: | :---: |
| **ZERITH 2.5 · LIBERO** | LIBERO 推理与评测 | [🤗 Hugging Face](https://huggingface.co/OpenZerith/ZERITH-2.5-LIBERO) | [指南](docs/simulation/libero_zh-CN.md) |
| **ZERITH 2.5 · LIBERO-Plus** | LIBERO-Plus 推理与评测 | [🤗 Hugging Face](https://huggingface.co/OpenZerith/ZERITH-2.5-LIBERO-Plus) | [指南](docs/simulation/libero-plus_zh-CN.md) |
| **ZERITH 2.5 · MetaWorld** | MetaWorld MT50 推理与评测 | [🤗 Hugging Face](https://huggingface.co/OpenZerith/ZERITH-2.5-MetaWorld) | [指南](docs/simulation/metaworld_zh-CN.md) |
| **ZERITH 2.5 · H1** | H1 中期训练（mid-training）的 backbone 权重 | [🤗 Hugging Face](https://huggingface.co/OpenZerith/ZERITH-2.5-H1) | [指南](#h1-部署) |

上述 Hub 仓库需要访问权限。仿真策略采用 FP32 推理；所需编码器已包含在
Quick Start 的下载脚本中。

## 快速开始

以 **LIBERO-Plus** 为例，直接评测发布权重：从 `libero_spatial` 中运行任务
`0、30、60`，每个任务一个 episode，seed 固定为 0。无需训练或下载示范数据。
以下必需步骤均在本页给出，每个终端的命令都从源码仓库根目录执行。

需要 Linux、Conda、兼容 CUDA 12.8 PyTorch 的 NVIDIA 驱动，以及有权访问模型仓库的
Hugging Face 账号。发布策略采用 FP32，权重本身约 27 GB；加上编码器和运行时，
以下示例建议使用 80 GB 显存的 GPU。

### 1. 安装基础环境

Ubuntu/Debian 先安装系统依赖；以 root 登录时去掉 `sudo`：

```bash
sudo apt-get update
sudo apt-get install -y git curl

git clone https://github.com/inFpZero/ZERITH-2.5.git
cd ZERITH-2.5
sudo bash scripts/install_libero_system_deps.sh
conda create --override-channels -c conda-forge -n zerith-inference python=3.12 pip -y
conda activate zerith-inference
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r simulation/requirements-inference.txt
```

### 2. 一次性下载所有所需资产

在 `zerith-inference` 环境中运行：

```bash
bash scripts/download_libero_plus_assets.sh
```

脚本集中下载策略权重、Wan/VGGT 源码与权重、LIBERO-Plus 模拟器源码与资产，
并生成 40 条标准任务指令。所有文件保存至 `simulation/assets/`。
已有的 Hugging Face 登录或 `HF_TOKEN` 会自动复用，未登录时提示登录；
重复运行会复用已下载的内容，保留本地修改。

### 3. 配置独立的模拟器环境

```bash
python scripts/configure_libero.py --benchmark-root simulation/assets/LIBERO-plus \
  --config-dir simulation/assets/libero-plus-config \
  --asset-archive simulation/assets/libero-plus-download/assets.zip
conda create --override-channels -c conda-forge -n zerith-libero-plus python=3.10 pip -y
conda run --no-capture-output -n zerith-libero-plus python -m pip install 'setuptools<82'
conda run --no-capture-output -n zerith-libero-plus python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
conda run --no-capture-output -n zerith-libero-plus python -m pip install \
  -c simulation/constraints-libero-client.txt -r simulation/assets/LIBERO-plus/requirements.txt \
  -r simulation/assets/LIBERO-plus/extra_requirements.txt 'requests>=2.32'
conda run --no-capture-output -n zerith-libero-plus python -m pip install --no-deps -e simulation/assets/LIBERO-plus
```

配置工具校验仿真资产的 SHA256、整理压缩包内的目录层级，并生成独立的 LIBERO-Plus
配置。CPU PyTorch 只供模拟器依赖使用；模型推理继续运行在 GPU 环境中。

### 4. 启动策略服务

终端 1，从仓库根目录启动服务并保持运行：

```bash
conda run --no-capture-output -n zerith-inference --cwd simulation python -m zerith_sim.posttrain serve --artifact assets/policies/libero-plus \
  --base-instructions assets/libero-instructions.json \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

### 5. 运行快速评测

终端 2，进入同一源码仓库根目录后执行。健康检查会等待模型服务加载完成，
最多等待十分钟：

```bash
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/simulation/assets/libero-plus-config"
curl --fail --retry 120 --retry-delay 5 --retry-connrefused --retry-max-time 600 \
  http://127.0.0.1:8000/health
conda run --no-capture-output -n zerith-libero-plus --cwd simulation python -m zerith_sim.posttrain eval --bench libero-plus --benchmark-root assets/LIBERO-plus \
  --suite libero_spatial --task-ids 0 30 60 --episodes 1 --seed 0 \
  --server http://127.0.0.1:8000 --output results/libero-plus-quickstart
```

`/health` 应返回 `libero-plus`。结束时终端打印 **`EVAL_SUMMARY`**，并在
`simulation/results/libero-plus-quickstart/` 下生成：

| 文件 | 内容 |
| :--- | :--- |
| `episodes.jsonl` | 每次试验的任务 ID、episode ID、成功标记和执行步数 |
| `summary.json` | 所选任务、seed、完成次数、成功次数和成功率（0–1） |

这个三个 episode 的示例用于快速体验闭环执行，不代表完整基准分数。
完整评测与自行训练见 [LIBERO-Plus 完整流程](docs/simulation/libero-plus_zh-CN.md)。

## 仿真评测与训练

每个基准均提供两条完整流程：**直接评测发布权重**，或**准备数据并训练自己的策略后再评测**。

| 基准 | 完整流程 |
| :--- | :---: |
| LIBERO | [下载、评测、训练与导出](docs/simulation/libero_zh-CN.md) |
| LIBERO-Plus | [下载、评测、训练与导出](docs/simulation/libero-plus_zh-CN.md) |
| MetaWorld MT50 | [下载、评测、训练与导出](docs/simulation/metaworld_zh-CN.md) |

环境、共享编码器、缓存约定及多机训练参数集中在[仿真公共配置](docs/simulation_zh-CN.md)。

## H1 部署

发布的 H1 权重是 H1 中期训练（mid-training）的 backbone checkpoint，用于初始化和离线推理。
真机部署需验证过的 H1 推理包，以及单独安装的 SDK 和 IK。
安装、配置和启停见 [H1 部署指南](docs/deployment_zh-CN.md)。

## 文档

| 组件 / 工作流 | English | 简体中文 |
| :--- | :---: | :---: |
| 推理资产、微调与评测 | [Simulation guide](docs/simulation.md) | [仿真指南](docs/simulation_zh-CN.md) |
| H1 推理与机器人控制 | [Deployment guide](docs/deployment.md) | [部署指南](docs/deployment_zh-CN.md) |
| 源码贡献与检查 | [Contributing](CONTRIBUTING.md) | — |

源码组件：[simulation/](simulation/)、[公共客户端](deployment/shared/)、
[推理服务](deployment/inference/)和 [H1 控制端](deployment/robot/)。
安全问题请按 [SECURITY.md](SECURITY.md) 报告。

## 许可

项目代码采用 [Apache-2.0](LICENSE)。Zerith 自有模型权重采用
[CC BY-NC 4.0](MODEL_LICENSE.txt)，允许非商业使用、修改、微调及再分发，
按条款署名并保留相关声明；不授予商业使用权，也不要求公开修改权重或业务源码。

第三方组件保留各自条款，详见[源码许可范围](LICENSE_POLICY.md)、
[权重政策](MODEL_LICENSE_POLICY.md)和 [NOTICE](NOTICE)。

## 引用与致谢

使用 ZERITH 2.5 开展研究时，请参阅技术报告：

**[ZERITH 2.5: From Predictive Power to Policy Learnability](https://zerith-25-review.kindpetal17.chatgpt.site/reader.html)** — ZERITH Z-Lab，Zerith Robotics。

```bibtex
@techreport{han2026zerith25,
  title = {ZERITH 2.5: From Predictive Power to Policy Learnability},
  author = {Chenxi Han and Xinmiao Yu and Epoch and Orion and Luke and Bin Qian and Dusk and Lyra and Cal and Alvin and Hongbo Duan and Alex and Jiahe Han and Tao Zhang and Guoping Pan and Houde Liu and Yi Cheng},
  institution = {ZERITH Z-Lab},
  year = {2026},
  url = {https://zerith-25-review.kindpetal17.chatgpt.site/assets/ZERITH-2.5-v1.0.pdf}
}
```

本项目基于 [Wan2.2](https://github.com/Wan-Video/Wan2.2)、
[VGGT](https://github.com/facebookresearch/vggt)和
[UMT5](https://huggingface.co/google/umt5-xxl)构建，并感谢 LIBERO、LIBERO-Plus
和 MetaWorld 社区。第三方署名保留在 [NOTICE](NOTICE) 及各组件声明中。
