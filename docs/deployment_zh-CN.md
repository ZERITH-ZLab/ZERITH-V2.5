# H1 部署

[项目概览](../README_zh-CN.md) · [English](deployment.md)

GPU 主机运行模型推理；H1 主机运行相机、状态采集、FK/IK 和 SDK 控制。

## 1. 分主机安装

**GPU 主机：** Linux、Python 3.10 或更新版本，以及支持 BF16 的 CUDA GPU。
从仓库根目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ./deployment/shared ./deployment/inference
cd deployment
cp configs/server.example.json configs/server.local.json
cp configs/visual_assets.example.json configs/visual_assets.local.json
```

**H1 主机：** 使用与运动、相机 SDK 匹配的 Python 环境。从仓库根目录执行：

```bash
python -m pip install ./deployment/shared ./deployment/robot
cd deployment
cp configs/h1.example.json configs/h1.local.json
zerith robot
```

`zerith robot` 默认仅做离线检查，不连接硬件。

## 2. 准备资产与配置

### 模型与编码器

[H1 发布权重](https://huggingface.co/OpenZerith/ZERITH-2.5-H1)是 H1 中期训练（mid-training）的 backbone checkpoint，用于初始化和离线推理。
其坐标、腰部单位及夹爪统计尚不符合真机部署约定，**不能直接用于真机**。
真机部署需另行取得提供方验证过的 H1 推理包。

如需下载该 backbone 权重，在仓库根目录执行：

```bash
hf auth login
python scripts/download_model.py --target h1 --output deployment/assets/h1-initializer
```

以下命令均在 `deployment/` 目录执行。将兼容推理包放入 `assets/policy`；
若提供方给出的是兼容训练导出，转换为推理包：

```bash
zerith prepare --source /path/to/exported-policy --output assets/policy
```

输出目录必须尚不存在。对已验证符合 H1 部署约定的初始化权重，还需提供部署元数据：

```bash
zerith prepare --source /path/to/verified-initializer --output assets/policy \
  --deployment-metadata /path/to/provider-verified-deployment.json
```

元数据由模型提供方提供，包含 `action_schema`、`camera_schema` 和 `vae_encoder`。

按模型的 `dependencies.json` 获取匹配的 Wan VAE、Wan/VGGT 源码、VGGT 权重及
UMT5 文本库，放置如下：

```text
assets/
  policy/model.safetensors
  policy/manifest.json
  wan/Wan2.2_VAE.pth
  vggt/config.json
  vggt/model.safetensors
  text/tasks.json
  text/text_embeddings.npy
  text/text_manifest.json
external/
  Wan2.2/
  vggt/
```

H1 的 `prompt` 必须与 `assets/text/tasks.json` 中的某条指令完全一致，并有对应 embedding。

### 本地配置

编辑安装时复制的三个配置文件。相对路径均按 `deployment/` 解析：

| 配置文件 | 必填内容 |
| --- | --- |
| [server.local.json](../deployment/configs/server.example.json) | 模型、视觉与文本资产路径，GPU、监听地址和端口 |
| [visual_assets.local.json](../deployment/configs/visual_assets.example.json) | Wan/VGGT 源码与权重路径，以及提供方给出的校验值 |
| [h1.local.json](../deployment/configs/h1.example.json) | 服务地址、准确的 `prompt`、SDK、规划器和相机配置；调试时保持 `commissioned: false` |

源码指纹可用下列命令计算：

```bash
zerith source-hash external/Wan2.2/wan
zerith source-hash external/vggt/vggt
```

### H1 SDK 与规划器

H1 主机需单独安装与固件及 Python ABI 匹配的
[Zerith SDK](https://github.com/inFpZero/zerith_public_sdk)、原生 IK、H1 URDF 和碰撞资产。
示例配置使用 `H1_SDK_1.3.9`：

| 字段 | 设置 |
| --- | --- |
| `sdk_root` | `external/zerith_public_sdk/H1_SDK_1.3.9/h1_sdk_v1.3.9_python3.10` |
| `camera_sdk_root` | `external/zerith_public_sdk/H1_SDK_1.3.9/camera_sdk_python` |
| `planner_source` | 规划器提供方的 Python 源码目录 |
| `planner_factory` | 规划器入口，示例为 `vla_source:build_planner` |

先运行 SDK 示例确认相机和电机可用。自定义规划器需符合[原生 IK 接口](../deployment/robot/src/zerith/h1/native_ik.py)。

## 3. 检查与启动

**GPU 主机：** 校验资产并启动服务，保持终端运行：

```bash
zerith doctor --role server --config configs/server.local.json --verify-assets
bash scripts/start_server.sh --config configs/server.local.json
```

**H1 主机：** 检查配置和服务连接：

```bash
zerith doctor --role h1 --config configs/h1.local.json
zerith robot --check-server --config configs/h1.local.json
```

`doctor` 不连接硬件。服务绑定非回环地址时，两端必须设置相同的 `ZERITH_API_KEY`
环境变量，长度至少 32 字符，并通过 SSH 隧道或 TLS 连接。

完成 SDK、坐标与关节、相机和急停验收后，设置 `commissioned: true`，
在交互终端启动控制器：

```bash
bash scripts/start_h1.sh --config configs/h1.local.json --execute
```

输入 `EXECUTE` 允许 SDK 初始化，检查姿态和相机视图，再按 Enter 开始推理。
初始化可能使机器人移动，确保只有一个控制器发送指令。

先停控制器，再停推理服务。在各自主机的另一个终端执行：

```bash
bash scripts/stop_h1.sh
bash scripts/stop_server.sh
```

也可在运行终端按 Ctrl+C。正常 SDK 反初始化可能使硬件移动；紧急情况使用实体急停。
主机不支持脚本停止时，使用原终端的 Ctrl+C。

## 参考：坐标与 HTTP

`state23` 与动作使用以下布局：

| 索引 | 含义 | 表示方式 |
| --- | --- | --- |
| `0:3`、`10:13` | 左/右手位置 | `body_yaw_link` 坐标系下的米 |
| `3:9`、`13:19` | 左/右手旋转 | 旋转矩阵前两列，按列展开 |
| `9`、`19` | 左/右夹爪 | 状态 0–1；命令 0–1.5 |
| `20:23` | 腰部升降、俯仰、偏航 | 米、弧度、弧度 |

相机顺序固定为头部、左腕、右腕：`cam_high`、`cam_left_wrist`、`cam_right_wrist`。
每次执行 4 步动作，名义控制频率为 30 Hz，实际频率受推理与 IK 延迟影响。

| 端点 | 用途 |
| --- | --- |
| `GET /health` | 服务元信息和可用任务 |
| `POST /reset` | 设置 `prompt` 与 `seed`，开始或重置会话 |
| `POST /infer` | 输入 NPZ 观测，返回动作块 |
| `POST /release` | 释放会话 |

自定义客户端建议使用 [PolicyClient](../deployment/shared/src/zerith/client.py)，
两端安装相同版本的共享包。输入包含 `images`（RGB uint8 `[T,3,H,W,3]`）、
`state23`（float32 `[23]`）和 `valid23`（bool `[23]`）；首次请求一帧，后续四帧。
返回 `commands`（float32 `[4,23]`）与 `valid`（bool `[4,23]`）。

### 会话与恢复

每个服务同时只处理一个会话，结束时调用 `close()` 释放。推理超时后先 reset，
再发送新观测；客户端丢失且无法释放会话时，重启服务。
IK 失败后检查机器人状态，再按 Enter 重新规划；SDK 部分发送失败需排查故障后重启控制器。
