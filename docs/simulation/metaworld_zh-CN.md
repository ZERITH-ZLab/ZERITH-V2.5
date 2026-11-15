# MetaWorld MT50：评测与训练

[公共配置](../simulation_zh-CN.md) · [项目主页](../../README_zh-CN.md) · [English](metaworld.md)

可以直接使用发布的训练好权重评测，也可以准备演示数据并训练自己的策略。
直接评测执行第 1–4 节；自行训练先完成第 1–2 节，再按第 5 节执行。
先完成[公共环境与编码器配置](../simulation_zh-CN.md)一次，后续无需重复下载共享编码器。
下文命令均从源码仓库根目录执行。

## 1. 下载本基准所需内容

在 `zerith-inference` 环境中下载发布权重及基准资产：

```bash
conda activate zerith-inference
hf auth login
python scripts/download_model.py --target metaworld --output simulation/assets/policies/metaworld
```

仅自行训练时，还需下载下面的演示数据；直接评测跳过此块：

```bash
hf download MINT-SJTU/Evo1_MetaWorld_Dataset --repo-type dataset \
  --revision 612c4e83d0e3b4e0be5227970bd8f5f07fc0c488 --local-dir simulation/data/raw/metaworld
```

数据根目录应包含 `meta/info.json`。准备流程读取发布者的 train split，保留其原始 episode ID。

此演示数据由第三方 MINT-SJTU 提供。

## 2. 配置独立的模拟器环境

```bash
conda create --override-channels -c conda-forge -n zerith-metaworld python=3.11 pip -y
conda run --no-capture-output -n zerith-metaworld python -m pip install metaworld==3.1.1 mujoco==3.3.0 'numpy>=1.26' 'requests>=2.32'
```

此环境只运行 MT50 客户端，不加载模型权重。

## 3. 启动发布权重的策略服务

终端 1，在仓库根目录执行并保持运行：

```bash
conda run --no-capture-output -n zerith-inference --cwd simulation python -m zerith_sim.posttrain serve --artifact assets/policies/metaworld \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

## 4. 完整基准评测

终端 2，进入同一仓库根目录执行：

```bash
export MUJOCO_GL=egl
curl --fail --retry 120 --retry-delay 5 --retry-connrefused --retry-max-time 600 \
  http://127.0.0.1:8000/health
conda run --no-capture-output -n zerith-metaworld --cwd simulation python -m zerith_sim.posttrain eval --bench metaworld --episodes 50 --seed 0 \
  --server http://127.0.0.1:8000 --output results/metaworld
```

此命令覆盖 MT50 的 50 个任务，每个任务 50 个目标，horizon 500，无 warmup。
输出目录为 `simulation/results/metaworld/`。

每个输出目录包含 `episodes.jsonl` 和完成后的 `summary.json`；成功率取值 0–1。
小规模检查与完整 benchmark 的评测范围不同，不应将其分数混用。

## 5. 自行训练、导出与评测

以下示例从本基准的兼容发布权重初始化，在自己的训练流程中重新训练策略。
也可以将 `--checkpoint` 换成架构匹配的自有初始化权重。
先完成[公共训练环境配置](../simulation_zh-CN.md#环境)，并下载第 1 节的演示数据。

### 准备数据

```bash
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain prepare --bench metaworld \
  --source data/raw/metaworld --output data/metaworld \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0
```

首次小规模检查可加 `--limit 2` 并换用独立的输出目录；正式训练使用完整准备结果。
中断后在相同命令加 `--resume`，更改数据或编码器时使用新目录。

### 训练

```bash
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain train --bench metaworld --data data/metaworld \
  --checkpoint assets/policies/metaworld/model.safetensors --output runs/metaworld-custom \
  --gpus 8 --batch-size 10 --steps 50000
```

示例为每节点 8 卡、每卡 batch 10。按资源调整，并参考[公共训练参数](../simulation_zh-CN.md#训练参数)
了解全局 batch、多机和恢复设置。

### 导出与服务

训练结束后选择最新导出的模型权重，生成独立部署目录：

```bash
ZERITH_CHECKPOINT=$(conda run -n zerith-train python -c "from pathlib import Path; print(max(Path('simulation/runs/metaworld-custom/checkpoints').glob('model-step-*.safetensors'), key=lambda p: p.stat().st_mtime).resolve())")
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain export --run runs/metaworld-custom \
  --checkpoint "$ZERITH_CHECKPOINT" --output artifacts/metaworld-custom
```

先停止第 3 节的服务，再用新权重启动：

```bash
conda run --no-capture-output -n zerith-inference --cwd simulation python -m zerith_sim.posttrain serve --artifact artifacts/metaworld-custom \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

在另一个终端重新运行第 4 节客户端，或用下面的命令将结果保存在单独目录：

```bash
export MUJOCO_GL=egl
curl --fail --retry 120 --retry-delay 5 --retry-connrefused --retry-max-time 600 \
  http://127.0.0.1:8000/health
conda run --no-capture-output -n zerith-metaworld --cwd simulation python -m zerith_sim.posttrain eval --bench metaworld --episodes 50 --seed 0 \
  --server http://127.0.0.1:8000 --output results/metaworld-custom
```
