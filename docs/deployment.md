# H1 Deployment

[Project overview](../README.md) · [简体中文](deployment_zh-CN.md)

The GPU host runs model inference. The H1 host handles cameras, state capture,
FK/IK and SDK control.

## 1. Install by Host

**GPU host:** Linux, Python 3.10 or newer and a CUDA GPU with BF16 support.
Run from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ./deployment/shared ./deployment/inference
cd deployment
cp configs/server.example.json configs/server.local.json
cp configs/visual_assets.example.json configs/visual_assets.local.json
```

**H1 host:** Use a Python environment matching the motion and camera SDK ABI.
Run from the repository root:

```bash
python -m pip install ./deployment/shared ./deployment/robot
cd deployment
cp configs/h1.example.json configs/h1.local.json
zerith robot
```

`zerith robot` performs an offline check by default without connecting to hardware.

## 2. Prepare Assets and Configuration

### Model and Encoders

The [released H1 weights](https://huggingface.co/OpenZerith/ZERITH-2.5-H1) are
the backbone checkpoint from H1 mid-training, intended for initialization and
offline inference. Its coordinates, waist units and
gripper statistics do not yet match the deployment contract, so **it cannot be
used directly on a robot**. Obtain a provider-verified H1 inference bundle for deployment.

To download the backbone checkpoint, run from the repository root:

```bash
hf auth login
python scripts/download_model.py --target h1 --output deployment/assets/h1-initializer
```

Run all commands below from `deployment/`. Place a compatible inference bundle
at `assets/policy`. If the provider supplies a compatible training export, convert it:

```bash
zerith prepare --source /path/to/exported-policy --output assets/policy
```

The output directory must not exist. An initializer already verified for H1
deployment also requires deployment metadata:

```bash
zerith prepare --source /path/to/verified-initializer --output assets/policy \
  --deployment-metadata /path/to/provider-verified-deployment.json
```

Obtain this metadata from the model provider. It contains `action_schema`,
`camera_schema` and `vae_encoder`.

Use the model's `dependencies.json` to obtain the matching Wan VAE, Wan/VGGT
sources, VGGT weights and UMT5 text bank. Arrange them as follows:

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

The H1 `prompt` must exactly match a string in `assets/text/tasks.json` with a
corresponding embedding.

### Local Configuration

Edit the three configuration files copied during installation. Relative paths
resolve from `deployment/`:

| Configuration | Required values |
| --- | --- |
| [server.local.json](../deployment/configs/server.example.json) | Model, visual/text asset paths, GPU, bind address and port |
| [visual_assets.local.json](../deployment/configs/visual_assets.example.json) | Wan/VGGT source and weight paths, plus provider-supplied verification hashes |
| [h1.local.json](../deployment/configs/h1.example.json) | Server URL, exact `prompt`, SDK, planner and cameras; keep `commissioned: false` during setup |

Compute source fingerprints with:

```bash
zerith source-hash external/Wan2.2/wan
zerith source-hash external/vggt/vggt
```

### H1 SDK and Planner

On the H1 host, install the [Zerith SDK](https://github.com/inFpZero/zerith_public_sdk)
matching the firmware and Python ABI, plus native IK, the H1 URDF and collision assets.
The example uses `H1_SDK_1.3.9`:

| Field | Value |
| --- | --- |
| `sdk_root` | `external/zerith_public_sdk/H1_SDK_1.3.9/h1_sdk_v1.3.9_python3.10` |
| `camera_sdk_root` | `external/zerith_public_sdk/H1_SDK_1.3.9/camera_sdk_python` |
| `planner_source` | The planner provider's Python source directory |
| `planner_factory` | Planner entry point, for example `vla_source:build_planner` |

Check camera and motor access using the SDK examples first. Custom planners must
follow the [native IK interface](../deployment/robot/src/zerith/h1/native_ik.py).

## 3. Check and Start

**GPU host:** Verify assets and start the service, leaving the terminal running:

```bash
zerith doctor --role server --config configs/server.local.json --verify-assets
bash scripts/start_server.sh --config configs/server.local.json
```

**H1 host:** Check configuration and the server connection:

```bash
zerith doctor --role h1 --config configs/h1.local.json
zerith robot --check-server --config configs/h1.local.json
```

`doctor` does not connect to hardware. Non-loopback serving requires the same
`ZERITH_API_KEY` environment variable, at least 32 characters long, on both hosts.
Connect through an SSH tunnel or TLS.

After commissioning the SDK, coordinates and joints, cameras and emergency stop,
set `commissioned: true` and start the controller from an interactive terminal:

```bash
bash scripts/start_h1.sh --config configs/h1.local.json --execute
```

Type `EXECUTE` to allow SDK initialization, check posture and camera views, then
press Enter to start inference. Initialization can move the robot. Allow only one
controller to send commands.

Stop the controller before the inference service. In another terminal on each host:

```bash
bash scripts/stop_h1.sh
bash scripts/stop_server.sh
```

You can also press Ctrl+C in the running terminal. Normal SDK deinitialization
can move hardware; use the physical emergency stop in an emergency. If scripted
stopping is unsupported on the host, use Ctrl+C in the original terminal.

## Reference: Coordinates and HTTP

`state23` and actions use this layout:

| Indices | Meaning | Representation |
| --- | --- | --- |
| `0:3`, `10:13` | Left/right hand position | Metres in `body_yaw_link` |
| `3:9`, `13:19` | Left/right hand rotation | First two rotation-matrix columns, flattened column by column |
| `9`, `19` | Left/right gripper | State 0–1; command 0–1.5 |
| `20:23` | Waist lift, pitch, yaw | Metres, radians, radians |

Camera order is head, left wrist, right wrist: `cam_high`, `cam_left_wrist`,
`cam_right_wrist`. Execute four actions per plan at a nominal 30 Hz; inference
and IK latency affect the actual rate.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service metadata and available tasks |
| `POST /reset` | Set `prompt` and `seed`; begin or reset a session |
| `POST /infer` | NPZ observation in, action chunk out |
| `POST /release` | Release the session |

For custom clients, use [PolicyClient](../deployment/shared/src/zerith/client.py)
and matching shared-package versions on both hosts. Inputs are `images` (RGB
uint8 `[T,3,H,W,3]`), `state23` (float32 `[23]`) and `valid23` (bool `[23]`), with
one frame initially and four thereafter. Outputs are `commands` (float32 `[4,23]`)
and `valid` (bool `[4,23]`).

### Sessions and Recovery

Each service handles one session at a time; call `close()` when finished. After
an inference timeout, reset before sending fresh observations. If the owning
client is lost and cannot release the session, restart the service.
After an IK failure, inspect the robot before pressing Enter to replan. A partial
SDK send requires fault investigation and a controller restart.
