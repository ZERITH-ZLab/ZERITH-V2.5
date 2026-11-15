# MetaWorld MT50: Evaluation and Training

[Shared setup](../simulation.md) · [Project homepage](../../README.md) · [简体中文](metaworld_zh-CN.md)

Evaluate our trained checkpoint directly, or prepare demonstrations and train your
own policy. Follow Sections 1–4 for direct evaluation. For your own training,
complete Sections 1–2 and then follow Section 5. Complete the
[shared environment and encoder setup](../simulation.md) once; shared encoders need no
additional download. All commands below start at the source repository root.

## 1. Download Benchmark Assets

In `zerith-inference`, download the released checkpoint and benchmark assets:

```bash
conda activate zerith-inference
hf auth login
python scripts/download_model.py --target metaworld --output simulation/assets/policies/metaworld
```

Only for training your own policy, also download the demonstrations below.
Skip this block for released-checkpoint evaluation:

```bash
hf download MINT-SJTU/Evo1_MetaWorld_Dataset --repo-type dataset \
  --revision 612c4e83d0e3b4e0be5227970bd8f5f07fc0c488 --local-dir simulation/data/raw/metaworld
```

The dataset root should contain `meta/info.json`. Preparation reads the publisher's
train split and retains its original episode IDs.

The demonstration dataset is provided by MINT-SJTU, a third party.

## 2. Set Up the Simulator Environment

```bash
conda create --override-channels -c conda-forge -n zerith-metaworld python=3.11 pip -y
conda run --no-capture-output -n zerith-metaworld python -m pip install metaworld==3.1.1 mujoco==3.3.0 'numpy>=1.26' 'requests>=2.32'
```

This environment runs only the MT50 client and does not load policy weights.

## 3. Serve the Released Checkpoint

In terminal 1, run from the source root and leave the process running:

```bash
conda run --no-capture-output -n zerith-inference --cwd simulation python -m zerith_sim.posttrain serve --artifact assets/policies/metaworld \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

## 4. Run the Complete Evaluation

In terminal 2, enter the same source root and run:

```bash
export MUJOCO_GL=egl
curl --fail --retry 120 --retry-delay 5 --retry-connrefused --retry-max-time 600 \
  http://127.0.0.1:8000/health
conda run --no-capture-output -n zerith-metaworld --cwd simulation python -m zerith_sim.posttrain eval --bench metaworld --episodes 50 --seed 0 \
  --server http://127.0.0.1:8000 --output results/metaworld
```

This covers all 50 MT50 tasks, 50 goals per task, horizon 500, without warmup.
Outputs are saved under `simulation/results/metaworld/`.

Each output directory contains `episodes.jsonl` and, on completion, `summary.json`;
success rates are on a 0–1 scale. Small execution checks and full benchmark runs
cover different evaluation sets and should not be compared as the same score.

## 5. Train, Export and Evaluate Your Own Policy

The example initializes from the benchmark's compatible released checkpoint and
trains a new policy. You may substitute your own architecture-compatible
`--checkpoint`. Complete the [shared training environment](../simulation.md#environment)
and download the demonstrations from Section 1 first.

### Prepare Demonstrations

```bash
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain prepare --bench metaworld \
  --source data/raw/metaworld --output data/metaworld \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0
```

For an initial preparation check, add `--limit 2` and use a separate output
directory; use the complete preparation for training. Resume matching interrupted
preparation with `--resume`, and use a new output after changing data or encoders.

### Train

```bash
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain train --bench metaworld --data data/metaworld \
  --checkpoint assets/policies/metaworld/model.safetensors --output runs/metaworld-custom \
  --gpus 8 --batch-size 10 --steps 50000
```

The example uses eight GPUs per node and batch 10 per GPU. Adjust for your
resources; see [shared training settings](../simulation.md#training-settings) for global
batch, distributed runs and recovery.

### Export and Serve

After training, select the latest exported model weights and create a separate
inference artifact:

```bash
ZERITH_CHECKPOINT=$(conda run -n zerith-train python -c "from pathlib import Path; print(max(Path('simulation/runs/metaworld-custom/checkpoints').glob('model-step-*.safetensors'), key=lambda p: p.stat().st_mtime).resolve())")
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain export --run runs/metaworld-custom \
  --checkpoint "$ZERITH_CHECKPOINT" --output artifacts/metaworld-custom
```

Stop the service from Section 3, then start one using your new artifact:

```bash
conda run --no-capture-output -n zerith-inference --cwd simulation python -m zerith_sim.posttrain serve --artifact artifacts/metaworld-custom \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

In a second terminal, run the client again and keep the results separate:

```bash
export MUJOCO_GL=egl
curl --fail --retry 120 --retry-delay 5 --retry-connrefused --retry-max-time 600 \
  http://127.0.0.1:8000/health
conda run --no-capture-output -n zerith-metaworld --cwd simulation python -m zerith_sim.posttrain eval --bench metaworld --episodes 50 --seed 0 \
  --server http://127.0.0.1:8000 --output results/metaworld-custom
```
