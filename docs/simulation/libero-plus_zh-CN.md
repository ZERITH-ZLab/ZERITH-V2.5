# LIBERO-Plus：评测与训练

[公共配置](../simulation_zh-CN.md) · [项目主页](../../README_zh-CN.md) · [English](libero-plus.md)

可以直接使用发布的训练好权重评测，也可以准备演示数据并训练自己的策略。
直接评测执行第 1–4 节；自行训练先完成第 1–2 节，再按第 5 节执行。
先完成[公共环境与编码器配置](../simulation_zh-CN.md)一次，后续无需重复下载共享编码器。
下文命令均从源码仓库根目录执行。
已完成主页 Quick Start 时，复用已下载的资产与模拟器环境，无需重复 clone 或创建环境。

## 1. 下载本基准所需内容

在 `zerith-inference` 环境中下载发布权重及基准资产：

```bash
conda activate zerith-inference
hf auth login
python scripts/download_model.py --target libero-plus --output simulation/assets/policies/libero-plus
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git simulation/assets/LIBERO
git -C simulation/assets/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
git clone https://github.com/sylvestf/LIBERO-plus.git simulation/assets/LIBERO-plus
git -C simulation/assets/LIBERO-plus checkout 4976dc30028e805ff8094b55501d532c48fec182
hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset \
  --revision dd2bd61b7d9a6fef1abc52d606e983b41886a149 --local-dir simulation/assets/libero-plus-download
python scripts/export_libero_instructions.py --libero-root simulation/assets/LIBERO \
  --output simulation/assets/libero-instructions.json
```

仅自行训练时，还需下载下面的演示数据；直接评测跳过此块：

```bash
hf download Sylvest/libero_plus_lerobot --repo-type dataset \
  --revision 22c57433fef692b5b9ecc0795344daac7fa867a5 --local-dir simulation/data/raw/libero-plus
```

数据根目录应包含 `meta/info.json`。准备流程读取发布者的 train split，保留其原始 episode ID。

## 2. 配置独立的模拟器环境

Ubuntu/Debian 需要 EGL/OpenGL 与 Expat、Fontconfig；LIBERO-Plus 另需 ImageMagick/Wand。
以下命令以 root 身份运行时去掉 `sudo`：

```bash
sudo bash scripts/install_libero_system_deps.sh
```

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

模拟器与模型使用不同环境。配置工具生成本基准独立的配置路径，避免与其他 LIBERO 安装冲突。

## 3. 启动发布权重的策略服务

终端 1，在仓库根目录执行并保持运行：

```bash
conda run --no-capture-output -n zerith-inference --cwd simulation python -m zerith_sim.posttrain serve --artifact assets/policies/libero-plus \
  --base-instructions assets/libero-instructions.json \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

## 4. 完整基准评测

终端 2，进入同一仓库根目录执行：

```bash
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/simulation/assets/libero-plus-config"
curl --fail --retry 120 --retry-delay 5 --retry-connrefused --retry-max-time 600 \
  http://127.0.0.1:8000/health
for suite in libero_spatial libero_object libero_goal libero_10; do
  conda run --no-capture-output -n zerith-libero-plus --cwd simulation python -m zerith_sim.posttrain eval --bench libero-plus --benchmark-root assets/LIBERO-plus \
    --suite "$suite" --episodes 1 --seed 0 \
    --server http://127.0.0.1:8000 --output results/libero-plus-"$suite"
done
```

上面的循环运行四套任务，每个任务 1 个 episode。每套的结果保存在
`simulation/results/libero-plus-<suite>/`。

若只想快速确认闭环执行，用 `--suite libero_spatial --task-ids 0 30 60 --episodes 1`
替代完整循环；首页的 [Quick Start](../../README_zh-CN.md#快速开始) 已给出全部命令。

每个输出目录包含 `episodes.jsonl` 和完成后的 `summary.json`；成功率取值 0–1。
小规模检查与完整 benchmark 的评测范围不同，不应将其分数混用。

## 5. 自行训练、导出与评测

以下示例从本基准的兼容发布权重初始化，在自己的训练流程中重新训练策略。
也可以将 `--checkpoint` 换成架构匹配的自有初始化权重。
先完成[公共训练环境配置](../simulation_zh-CN.md#环境)，并下载第 1 节的演示数据。

### 准备数据

```bash
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain prepare --bench libero-plus \
  --source data/raw/libero-plus --output data/libero-plus \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0
```

首次小规模检查可加 `--limit 2` 并换用独立的输出目录；正式训练使用完整准备结果。
中断后在相同命令加 `--resume`，更改数据或编码器时使用新目录。

### 训练

```bash
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain train --bench libero-plus --data data/libero-plus \
  --checkpoint assets/policies/libero-plus/model.safetensors --output runs/libero-plus-custom \
  --gpus 8 --batch-size 10 --steps 30000
```

示例为每节点 8 卡、每卡 batch 10。按资源调整，并参考[公共训练参数](../simulation_zh-CN.md#训练参数)
了解全局 batch、多机和恢复设置。

### 导出与服务

训练结束后选择最新导出的模型权重，生成独立部署目录：

```bash
ZERITH_CHECKPOINT=$(conda run -n zerith-train python -c "from pathlib import Path; print(max(Path('simulation/runs/libero-plus-custom/checkpoints').glob('model-step-*.safetensors'), key=lambda p: p.stat().st_mtime).resolve())")
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain export --run runs/libero-plus-custom \
  --checkpoint "$ZERITH_CHECKPOINT" --output artifacts/libero-plus-custom
```

先停止第 3 节的服务，再用新权重启动：

```bash
conda run --no-capture-output -n zerith-inference --cwd simulation python -m zerith_sim.posttrain serve --artifact artifacts/libero-plus-custom \
  --base-instructions assets/libero-instructions.json \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

在另一个终端重新运行第 4 节客户端，或用下面的命令将结果保存在单独目录：

```bash
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/simulation/assets/libero-plus-config"
curl --fail --retry 120 --retry-delay 5 --retry-connrefused --retry-max-time 600 \
  http://127.0.0.1:8000/health
for suite in libero_spatial libero_object libero_goal libero_10; do
  conda run --no-capture-output -n zerith-libero-plus --cwd simulation python -m zerith_sim.posttrain eval --bench libero-plus --benchmark-root assets/LIBERO-plus \
    --suite "$suite" --episodes 1 --seed 0 \
    --server http://127.0.0.1:8000 --output results/libero-plus-custom-"$suite"
done
```
