# LIBERO-Plus: Evaluation and Training

[Shared setup](../simulation.md) · [Project homepage](../../README.md) · [简体中文](libero-plus_zh-CN.md)

Evaluate our trained checkpoint directly, or prepare demonstrations and train your
own policy. Follow Sections 1–4 for direct evaluation. For your own training,
complete Sections 1–2 and then follow Section 5. Complete the
[shared environment and encoder setup](../simulation.md) once; shared encoders need no
additional download. All commands below start at the source repository root.
If you already completed the homepage quick start, reuse its downloads and
simulator environment rather than cloning or creating them again.

## 1. Download Benchmark Assets

In `zerith-inference`, download the released checkpoint and benchmark assets:

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

Only for training your own policy, also download the demonstrations below.
Skip this block for released-checkpoint evaluation:

```bash
hf download Sylvest/libero_plus_lerobot --repo-type dataset \
  --revision 22c57433fef692b5b9ecc0795344daac7fa867a5 --local-dir simulation/data/raw/libero-plus
```

The dataset root should contain `meta/info.json`. Preparation reads the publisher's
train split and retains its original episode IDs.

## 2. Set Up the Simulator Environment

On Ubuntu/Debian, install EGL/OpenGL, Expat and Fontconfig; LIBERO-Plus also
requires ImageMagick/Wand. Omit `sudo` when running as root:

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

The simulator and policy use separate environments. The configuration tool writes
benchmark-specific paths so other LIBERO installations remain independent.

## 3. Serve the Released Checkpoint

In terminal 1, run from the source root and leave the process running:

```bash
conda run --no-capture-output -n zerith-inference --cwd simulation python -m zerith_sim.posttrain serve --artifact assets/policies/libero-plus \
  --base-instructions assets/libero-instructions.json \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

## 4. Run the Complete Evaluation

In terminal 2, enter the same source root and run:

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

The loop evaluates all four suites, with 1 episode(s) per task. Each suite
writes to `simulation/results/libero-plus-<suite>/`.

For a small execution check, use `--suite libero_spatial --task-ids 0 30 60 --episodes 1`
instead of the full loop. The [homepage quick start](../../README.md#quick-start)
includes every required command.

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
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain prepare --bench libero-plus \
  --source data/raw/libero-plus --output data/libero-plus \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0
```

For an initial preparation check, add `--limit 2` and use a separate output
directory; use the complete preparation for training. Resume matching interrupted
preparation with `--resume`, and use a new output after changing data or encoders.

### Train

```bash
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain train --bench libero-plus --data data/libero-plus \
  --checkpoint assets/policies/libero-plus/model.safetensors --output runs/libero-plus-custom \
  --gpus 8 --batch-size 10 --steps 30000
```

The example uses eight GPUs per node and batch 10 per GPU. Adjust for your
resources; see [shared training settings](../simulation.md#training-settings) for global
batch, distributed runs and recovery.

### Export and Serve

After training, select the latest exported model weights and create a separate
inference artifact:

```bash
ZERITH_CHECKPOINT=$(conda run -n zerith-train python -c "from pathlib import Path; print(max(Path('simulation/runs/libero-plus-custom/checkpoints').glob('model-step-*.safetensors'), key=lambda p: p.stat().st_mtime).resolve())")
conda run --no-capture-output -n zerith-train --cwd simulation python -m zerith_sim.posttrain export --run runs/libero-plus-custom \
  --checkpoint "$ZERITH_CHECKPOINT" --output artifacts/libero-plus-custom
```

Stop the service from Section 3, then start one using your new artifact:

```bash
conda run --no-capture-output -n zerith-inference --cwd simulation python -m zerith_sim.posttrain serve --artifact artifacts/libero-plus-custom \
  --base-instructions assets/libero-instructions.json \
  --wan-repo assets/Wan2.2 --wan-model assets/Wan2.2-TI2V-5B \
  --vggt-repo assets/vggt --vggt-model assets/VGGT-1B --device cuda:0 --port 8000
```

In a second terminal, run the client again and keep the results separate:

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
