"""Run released policy inference without a simulator or robot connection.

By default the inputs are synthetic encoded features. Supply all four encoder
paths to exercise real Wan VAE, UMT5 and VGGT on generated RGB observations.
This checks execution and reset consistency, not task success or robot units.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ("simulation", "deployment/shared/src", "deployment/inference/src"):
    sys.path.insert(0, str(ROOT / relative))


def load_policy(target, root, device):
    import torch

    if target != "h1":
        from zerith_sim.models.zerith import load_deployment_artifact

        policy, manifest = load_deployment_artifact(
            root, device=device, dtype=torch.float32, verify_weight_sha256=True
        )
        if manifest["action_schema"]["benchmark"] != target:
            raise ValueError("Artifact benchmark differs from --target")
        return policy

    # Offline initializer loading retains the checkpoint's own codec buffers.
    # It does not produce a robot bundle or assert physical coordinate semantics.
    from safetensors.torch import load_file
    from zerith_core.actions.h1 import digest
    from zerith_core.models.zerith.config import ZerithConfig
    from zerith_core.models.zerith.runtime import ZerithTeacherPolicy

    manifest = json.loads((root / "manifest.json").read_text())
    from zerith.compat import canonical_identifier

    if canonical_identifier(manifest.get("format")) != "zerith-initializer-v1":
        raise ValueError("Expected an H1 initializer")
    weight_path = root / "model.safetensors"
    sha = hashlib.sha256()
    with weight_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            sha.update(chunk)
    if sha.hexdigest() != manifest["weights"]["sha256"]:
        raise ValueError("H1 weight SHA256 mismatch")
    config = json.loads((root / "config.json").read_text())
    targets = config["teacher_target_layers"]
    config["student_layers"] = len(targets)
    config["student_source_layers"] = targets
    config["reader_source_layers"] = [targets[i] for i in config["reader_kv_indices"]]
    config.pop("metaworld_native4", None)
    with torch.device("meta"):
        policy = ZerithTeacherPolicy(ZerithConfig.from_dict(config))
    weights = load_file(str(weight_path), device="cpu")
    weights = {
        (
            "world." + k.removeprefix("teacher_carrier.") if k.startswith("teacher_carrier.") else k
        ): v
        for k, v in weights.items()
    }
    policy.load_state_dict(weights, strict=True, assign=True)
    norm = json.loads((root / "norm.json").read_text())
    if bytes(policy.state_codec.stats_sha256.tolist()).hex() != digest(norm):
        raise ValueError("H1 normalization fingerprint mismatch")
    policy.world.wan.mark_official_weights_loaded()
    return policy.to(device=device).eval()


def state_input(target, device):
    import torch

    width = 23 if target == "h1" else 20
    state = torch.zeros(1, width, device=device)
    valid = torch.zeros_like(state, dtype=torch.bool)
    if target == "h1":
        state[:, [3, 7, 13, 17]] = 1  # Identity rotation6D for both arms.
        valid[:] = True
    elif target == "metaworld":
        state[:, :3] = torch.tensor([0, 0.6, 0.2], device=device)
        valid[:, [0, 1, 2, 9]] = True
    else:
        state[:, :3] = torch.tensor([0, 0, 0.2], device=device)
        state[:, [3, 7]] = 1
        valid[:, :10] = True
    return state, valid


def encoder_inputs(args, device):
    import numpy as np
    import torch
    from zerith_sim.vision.wan_assets import _load_text_encoder

    encoder = _load_text_encoder(
        Path(args.wan_model), Path(args.wan_repo), device, initialize_on_cpu=True
    )
    with torch.inference_mode():
        value = encoder([args.prompt], device)[0]
    if len(value) > 64:
        raise ValueError("Prompt exceeds the released text64 prefix")
    text = torch.zeros(1, 64, 4096, device=device, dtype=torch.float16)
    text[0, : len(value)] = value
    del encoder, value
    gc.collect()
    torch.cuda.empty_cache()
    if args.target == "h1":
        from zerith.contract import CAMERAS
        from zerith_core.vision.schema import CameraProfile
        from zerith_core.vision.streaming import (
            StreamingVGGTGeometryEncoder,
            StreamingWanVAEEncoder,
        )
        from zerith_core.vision.wan_vae import load_wan22_vae

        names = CAMERAS
    else:
        from zerith_sim.posttrain.recipes import BENCHMARKS
        from zerith_sim.posttrain.visual import StreamingVGGTGeometryEncoder, StreamingWanVAEEncoder
        from zerith_sim.vision import CameraProfile, load_wan22_vae

        names = BENCHMARKS[args.target]["cameras"]
    profile = CameraProfile(
        args.target, names, tuple(range(len(names))), ("front", "left", "right")
    )
    vae = load_wan22_vae(Path(args.wan_model) / "Wan2.2_VAE.pth", args.wan_repo, device)
    visual = StreamingWanVAEEncoder(vae, profile, device)
    geometry = StreamingVGGTGeometryEncoder(
        Path(args.vggt_repo), Path(args.vggt_model), profile, device
    )
    rng = np.random.default_rng(args.seed)
    images = [rng.integers(0, 256, (n, len(names), 480, 480, 3), dtype=np.uint8) for n in (1, 4)]
    if args.target == "metaworld":
        from metaworld_benchmark.contract import stored_image

        images = [np.stack([stored_image(frame[0]) for frame in x])[:, None] for x in images]

    def encode(index):
        from zerith_sim.posttrain.visual import policy_latent

        latent, spatial = visual.encode(images[index], names)
        latent = policy_latent(latent, torch.float32)
        geo, gv = geometry.encode(images[index], names)
        return latent, spatial, text, geo, gv

    def reset():
        visual.reset()
        geometry.reset()

    return encode, reset


def synthetic_inputs(args, device):
    import torch

    generator = torch.Generator(device=device).manual_seed(args.seed)
    cameras = 3 if args.target == "h1" else 1 if args.target == "metaworld" else 2
    spatial = torch.zeros(1, 42, 14, device=device, dtype=torch.bool)
    spatial[:, : 14 * cameras] = True
    gv = torch.zeros(1, 147, device=device, dtype=torch.bool)
    gv[:, : 49 * cameras] = True
    text = torch.zeros(1, 64, 4096, device=device, dtype=torch.float16)
    text[:, :8] = torch.randn(1, 8, 4096, generator=generator, device=device).half()
    values = []
    for _ in range(2):
        latent = torch.randn(1, 48, 42, 14, generator=generator, device=device) * spatial[:, None]
        geo = torch.randn(1, 147, 2048, generator=generator, device=device).half() * gv[..., None]
        values.append((latent, spatial, text, geo, gv))
    return lambda index: values[index], lambda: None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", required=True, choices=("libero-plus", "libero", "metaworld", "h1")
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="pick up the block")
    parser.add_argument("--output", type=Path, required=True)
    for name in ("wan-repo", "wan-model", "vggt-repo", "vggt-model"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    paths = [args.wan_repo, args.wan_model, args.vggt_repo, args.vggt_model]
    if any(paths) and not all(paths):
        parser.error("Supply all four encoder paths, or none for synthetic encoded inputs")
    import torch

    torch.set_num_threads(8)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("A compatible CUDA GPU is required for the full released model")
    torch.cuda.set_device(device)
    encode, reset = encoder_inputs(args, device) if all(paths) else synthetic_inputs(args, device)
    print("ENCODERS_READY", flush=True)
    policy = load_policy(args.target, args.artifact, device)
    print("POLICY_LOADED", flush=True)
    state, valid = state_input(args.target, device)
    dt = {"libero-plus": 0.05, "libero": 0.05, "metaworld": 0.0125, "h1": 1 / 30}[args.target]
    width = {"libero-plus": 7, "libero": 7, "metaworld": 4, "h1": 23}[args.target]
    policy.reset_inference(args.seed)
    calls, predictions = [], []
    for index in (0, 1, 0):
        if len(calls) == 2:
            reset()
            policy.reset_inference(args.seed)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            latent, spatial, text, geo, gv = encode(index)
            action, mask = policy.act_state_trajectory(
                latent,
                spatial,
                text,
                torch.tensor([dt], device=device),
                torch.tensor([4 * dt], device=device),
                state,
                valid,
                torch.zeros(1, device=device, dtype=torch.long),
                geometry_features=geo,
                geometry_valid=gv,
            )
        torch.cuda.synchronize(device)
        if action.shape != (1, 16, policy.config.action_dim) or mask.shape != action.shape:
            raise AssertionError("Prediction shape differs from predict16 contract")
        if not torch.isfinite(action).all() or not mask[..., :width].all():
            raise AssertionError(
                "Predictions are nonfinite or expected command columns are invalid"
            )
        if mask[..., width:].any() or (action[..., width:] != 0).any():
            raise AssertionError("Unused transport columns must be invalid and exactly zero")
        predictions.append(action.cpu())
        calls.append(
            {
                "seconds": time.perf_counter() - start,
                "prediction_shape": list(action.shape),
                "executed_shape": [4, width],
                "valid_columns": width,
            }
        )
        print("INFERENCE_PASS", len(calls), calls[-1], flush=True)
    torch.testing.assert_close(predictions[0], predictions[2], rtol=0, atol=0)
    result = dict(
        target=args.target,
        status="pass",
        input_mode="generated_rgb_and_real_encoders"
        if all(paths)
        else "synthetic_encoded_features",
        precision="float32",
        solver_steps=10,
        calls=calls,
        reset_exact_match=True,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
        simulator_used=False,
        robot_connected=False,
        physical_semantics_verified=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("SMOKE_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
