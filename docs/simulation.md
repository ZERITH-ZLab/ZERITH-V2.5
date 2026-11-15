# Shared Simulation Setup

[Project homepage](../README.md) · [简体中文](simulation_zh-CN.md)

This page contains only the environments, encoder assets, training settings and
cache conventions shared by the three benchmarks. Choose a separate workflow
below; each covers both released-checkpoint evaluation and training your own policy.

| Benchmark | Separate workflow |
| :--- | :--- |
| LIBERO | [Standard 40 tasks: checkpoints, data, evaluation and training](simulation/libero.md) |
| LIBERO-Plus | [Robustness suite: checkpoints, assets, evaluation and training](simulation/libero-plus.md) |
| MetaWorld | [MT50: checkpoints, data, evaluation and training](simulation/metaworld.md) |

All commands below start at the source repository root. `conda run --cwd simulation`
runs commands in the intended environment and directory; enter the source root
again when opening a new terminal.

## Environment

Use a separate GPU environment for the model service. Inference uses CUDA 12.8; the training reference uses CUDA 13.0.
Use a compatible NVIDIA driver for the selected build. Evaluating released weights requires
neither Megatron, demonstration datasets nor visual caches.

```bash
conda create --override-channels -c conda-forge -n zerith-inference python=3.12 pip -y
conda activate zerith-inference
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r simulation/requirements-inference.txt
```

For training your own policy, create an additional environment:

```bash
conda create --override-channels -c conda-forge -n zerith-train python=3.12 pip -y
conda run --no-capture-output -n zerith-train python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
conda run --no-capture-output -n zerith-train python -m pip install megatron-core==0.19.0
conda run --no-capture-output -n zerith-train python -m pip install -r simulation/requirements.txt
```

Install simulator clients in the separate environments specified by each benchmark
workflow. LIBERO and LIBERO-Plus also have separate installations and
`LIBERO_CONFIG_PATH` values.

## Required Assets

These encoder assets are shared by all three benchmarks and need to be downloaded
only once. Data preparation and inference must use identical versions. Run in
`zerith-inference`:

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

Each benchmark guide groups its checkpoint, simulator-asset and optional training
dataset downloads. Released checkpoints can be evaluated directly using FP32 inference.

## Cache and Action Conventions

Preparation produces normalization statistics and visual/language caches.
Use the same encoder versions for training and inference. Resume interrupted
preparation by adding `--resume`; use a new output directory after changing data,
encoders or configuration. Run only one preparation process per output, separate
from the source data and encoder directories.

LIBERO/Plus use native OSC7; MetaWorld uses clipped Native4. The policy predicts
16 actions and evaluation executes the first four.

## Training Settings

`--gpus` specifies GPUs per node; `--batch-size` specifies examples per GPU.
The global batch is **nodes × GPUs per node × batch size**. The launcher uses
`torchrun` with Megatron-FSDP and no gradient accumulation. Changing GPU
count changes the global batch; it does not automatically scale the learning
rate. Choose a batch that fits the available memory.

The generated recipes use these defaults:

| Setting | Default |
| --- | --- |
| LIBERO / LIBERO-Plus | 30,000 steps |
| MetaWorld | 50,000 steps |
| Base learning rate | `1e-5` |
| Wan / Reader backbone LR multiplier | 1 |
| Reader heads / geometry / G4D LR multiplier | 10 |
| AdamW | betas `(0.9, 0.95)`, weight decay `0.01` |
| Schedule | cosine, 5% warmup, minimum LR 10% of the initial value |
| Gradient clipping / precision | 1.0 / BF16 |
| Action gradient into Wan | zero for the first 5% of steps; linear ramp to 1 over the next 5% |
| Checkpoints | every 1,000 steps, retain two recovery points |

Set `--steps`, `--learning-rate`, `--batch-size` and `--workers` to override
the launcher defaults. Add `--dry-run` to print the generated configuration
and global batch without starting training.

### Distributed Training and Recovery

Add `--nodes N --node-rank R --master-addr TRAINING_HOST --master-port 29500`
to the selected benchmark's training command on every node. Ranks run from 0 to
N−1; data and initialization paths must agree across nodes, and the output directory
must be shared. `--gpus` is per node; four nodes with eight GPUs and per-GPU batch
10 give global batch 320.

To recover, keep topology, batch size, data and hyperparameters unchanged and add
`--resume` to the original command. The newest complete checkpoint restores the
optimizer, scheduler, random state and sampler cursor. Resume only from a run
directory you created or trust.

### Initialization and Export

Examples initialize from released weights. Use `--checkpoint` to select another
architecture-compatible initializer. Each benchmark guide includes data preparation,
training, export and serving commands.

## Evaluation Outputs

Runs write `episodes.jsonl`. Completed runs also write `summary.json` and print
`EVAL_SUMMARY`, including completed episodes, successes and success rate (0–1).
Each service process handles one episode at a time.

## Optional FlashAttention-3

PyTorch SDPA works by default. On Hopper GPUs, optionally build the
[custom extension](../simulation/fa2_custom_mask/) with a compatible CUDA toolkit
(including `nvcc`) and C++ compiler:

```bash
conda run --no-capture-output -n zerith-train python -m pip install ninja
conda run --no-capture-output -n zerith-train --cwd simulation python -c 'from fa2_custom_mask import prebuild_flash_attention3; print(prebuild_flash_attention3())'
```

A failed build falls back to SDPA. Set `ZERITH_REQUIRE_FLASH_ATTENTION3=1` to require the extension.

## Development checks

```bash
conda run --no-capture-output -n zerith-train --cwd simulation python -m pytest -q tests
```

Unit tests cover codecs, masking, sampling, model loading and artifact
roundtrips. Full training and closed-loop evaluation require the actual
model assets and simulator environments; unit tests do not establish
benchmark performance.
