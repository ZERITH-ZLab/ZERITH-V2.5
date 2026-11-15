"""Prepare a ZERITH 2.5 policy bundle from an exported deployment artifact."""

import argparse
import json
import shutil
from dataclasses import fields
from pathlib import Path

from .assets import read_json, sha256, verified_file
from .compat import LEGACY_POLICY_FORMAT, canonical_identifier
from .contract import CAMERA_PROFILE, CAMERAS, validate_h1_command_contract
from .encoder_contract import validate_encoder_contract

ZERITH_EXPORT_FORMAT = "zerith-policy-action20-linear-g4d-teacher"
LEGACY_EXPORT_FORMAT = LEGACY_POLICY_FORMAT
INITIALIZER_FORMAT = "zerith-initializer-v1"


def initializer_export(source, manifest, deployment_metadata):
    """Read an initializer without relabeling its normalization or coordinates."""
    codec = manifest.get("codec", {})
    if (
        codec.get("type") != "H1Command23Codec"
        or codec.get("state_dim") != 23
        or codec.get("action_dim") != 23
        or codec.get("embodiment_id") != 0
    ):
        raise ValueError("Expected an H1 Command23 initializer")
    # The published initializer declares these two fixed companion files.
    if manifest.get("config") != "config.json" or manifest.get("normalization") != "norm.json":
        raise ValueError("H1 initializer must supply config.json and norm.json")
    stats = read_json(source / "norm.json")
    try:
        validate_h1_command_contract(stats.get("command_contract"))
    except ValueError as error:
        raise ValueError(
            f"H1 initializer norm.json is incompatible with real-robot deployment: {error}. "
            "Obtain a checkpoint with verified deployment coordinates and waist units; "
            "do not relabel the existing statistics."
        ) from error
    if deployment_metadata is None:
        raise ValueError(
            "H1 initializer lacks deployment timing, camera and VAE metadata; "
            "supply provider-verified --deployment-metadata JSON with action_schema, "
            "camera_schema and vae_encoder."
        )
    metadata = read_json(deployment_metadata)
    config = read_json(source / "config.json")
    # Initializers serialize the active Teacher/Reader coordinates without the
    # redundant Student/Reader source-layer maps used by the deployment parser.
    taps, targets = config.get("kv_tap_layers", []), config.get("teacher_target_layers", [])
    if "student_source_layers" not in config and taps == list(range(len(targets))):
        config["student_layers"] = len(targets)
        config["student_source_layers"] = targets
    if "reader_source_layers" not in config:
        config["reader_source_layers"] = [targets[i] for i in config["reader_kv_indices"]]
    return dict(
        format=ZERITH_EXPORT_FORMAT,
        config=config,
        normalization_stats=stats,
        weights=manifest["weights"],
        action_schema=metadata["action_schema"],
        camera_schema=metadata["camera_schema"],
        vae_encoder=metadata["vae_encoder"],
    )


def prepare(source, destination, *, deployment_metadata=None):
    source, destination = Path(source), Path(destination)
    old = read_json(source / "manifest.json")
    initializer = (
        isinstance(old, dict) and canonical_identifier(old.get("format")) == INITIALIZER_FORMAT
    )
    if initializer:
        old = initializer_export(source, old, deployment_metadata)
    if not isinstance(old, dict) or canonical_identifier(old.get("format")) != ZERITH_EXPORT_FORMAT:
        raise ValueError(
            "Expected export format zerith-policy-action20-linear-g4d-teacher; "
            "provide the exported weights and manifest described in the ZERITH 2.5 README"
        )
    stats = old.get("normalization_stats")
    validate_h1_command_contract(stats.get("command_contract") if isinstance(stats, dict) else None)
    validate_encoder_contract(old.get("config"))
    from zerith_core.models.zerith.config import ZerithConfig

    allowed = {f.name for f in fields(ZerithConfig)}
    config = {k: v for k, v in old["config"].items() if k in allowed}
    if any("rtc" in k.lower() and bool(v) for k, v in old["config"].items()):
        raise ValueError("RTC-enabled checkpoints are incompatible with this policy loader")
    parsed = ZerithConfig.from_dict(config)
    if not parsed.h1_command23:
        raise ValueError("H1 Command23 weights required")
    schema = old["action_schema"]
    if (
        schema.get("width") != 23
        or schema.get("execute_rows") != [0, 1, 2, 3]
        or schema.get("primitive_dt") != 1 / 30
        or schema.get("video_dt") != 4 / 30
    ):
        raise ValueError("Checkpoint time/action schema mismatch")
    profiles = [
        profile
        for name, profile in old["camera_schema"]["profiles"].items()
        if canonical_identifier(name) == CAMERA_PROFILE
    ]
    if len(profiles) != 1:
        raise ValueError("Expected exactly one H1 camera profile")
    profile = profiles[0]

    if tuple(
        canonical_identifier(name) for name in profile["source_camera_names"]
    ) != CAMERAS or profile["source_to_canonical"] != [
        0,
        1,
        2,
    ]:
        raise ValueError("Checkpoint camera order mismatch")
    model = verified_file(source, "model.safetensors", old["weights"]["sha256"])
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model, destination / "model.safetensors")
    result = dict(
        format="zerith-h1-inference-v1",
        config=parsed.as_dict(),
        normalization_stats=old["normalization_stats"],
        weights_sha256=sha256(destination / "model.safetensors"),
        vae_sha256=old["vae_encoder"]["sha256"],
    )
    if initializer:
        # Retain the original tensor bytes; the loader maps the module prefix.
        result["weight_namespace"] = "training"
    (destination / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Export directory with weights and manifest.json"
    )
    parser.add_argument("--output", required=True, help="New directory for the inference bundle")
    parser.add_argument(
        "--deployment-metadata",
        help="provider-verified action_schema, camera_schema and vae_encoder JSON for an H1 initializer",
    )
    args = parser.parse_args(argv)
    prepare(args.source, args.output, deployment_metadata=args.deployment_metadata)
    print(f"Policy bundle prepared: {args.output}")


if __name__ == "__main__":
    main()
