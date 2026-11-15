"""Self-contained deployment artifacts: ``model.safetensors`` + ``manifest.json``."""

from __future__ import annotations

from zerith_sim.compat import LEGACY_POLICY_FORMAT, canonical_identifier

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from .config import ZerithConfig
from .model import ZerithPolicy
from .primitives import FP32Linear

ARTIFACT_FORMAT = "zerith-policy-action20-linear-g4d-teacher"
# Accept previously exported artifacts; all new exports use the ZERITH 2.5 name.
LEGACY_ARTIFACT_FORMAT = LEGACY_POLICY_FORMAT
WEIGHT_FILE = "model.safetensors"
MANIFEST_FILE = "manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    return json.loads(json.dumps(value, sort_keys=True))


def _move_policy(policy: ZerithPolicy, device, dtype) -> ZerithPolicy:
    """Cast weights to ``dtype`` except FP32 conditioners; buffers stay FP32."""
    fp32_ids = {
        id(parameter)
        for module in policy.modules() if isinstance(module, FP32Linear)
        for parameter in module.parameters(recurse=False)
    }
    for parameter in policy.parameters():
        target = (torch.float32 if id(parameter) in fp32_ids else
                  dtype if dtype is not None and parameter.is_floating_point()
                  else parameter.dtype)
        parameter.data = parameter.data.to(device=device, dtype=target)
    for module in policy.modules():
        for name, value in tuple(module._buffers.items()):
            if value is not None:
                module._buffers[name] = value.to(device=device)
    return policy


def export_deployment_artifact(
    policy: ZerithPolicy,
    output_dir: str | Path,
    *,
    action_schema: Any,
    camera_schema: Any,
    normalization_stats: Any,
    vae_encoder_provenance: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in policy.cpu().eval().state_dict().items()
    }
    weights = output / WEIGHT_FILE
    save_file(state, str(weights))
    manifest = {
        "format": ARTIFACT_FORMAT,
        "config": _json(ZerithConfig.from_dict(policy.config.as_dict())),
        "action_schema": _json(action_schema),
        "camera_schema": _json(camera_schema),
        "normalization_stats": _json(normalization_stats),
        "vae_encoder": _json(dict(vae_encoder_provenance)),
        "provenance": _json(dict(provenance or {})),
        "weights": {
            "file": WEIGHT_FILE,
            "sha256": sha256_file(weights),
            "bytes": weights.stat().st_size,
            "keys": len(state),
        },
    }
    (output / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_deployment_artifact(
    artifact_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: Optional[torch.dtype] = None,
    verify_weight_sha256: bool = False,
) -> Tuple[ZerithPolicy, dict]:
    root = Path(artifact_dir)
    manifest = json.loads((root / MANIFEST_FILE).read_text())
    if canonical_identifier(manifest.get("format")) != ARTIFACT_FORMAT:
        raise ValueError(f"unsupported ZERITH 2.5 deployment artifact format: {manifest.get('format')!r}")
    weights = root / WEIGHT_FILE
    if verify_weight_sha256 and sha256_file(weights) != manifest["weights"]["sha256"]:
        raise RuntimeError("deployment weight SHA-256 mismatch")
    # A released FP32 policy is ~27 GB: avoid a second initialized parameter copy.
    with torch.device('meta'):
        policy = ZerithPolicy(ZerithConfig.from_dict(manifest['config']))
    policy.load_state_dict(load_file(str(weights), device='cpu'), strict=True, assign=True)
    policy = _move_policy(policy, device, dtype).eval()
    return policy, manifest
