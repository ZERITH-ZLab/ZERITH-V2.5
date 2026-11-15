"""Initializer conversion preserves tensors and rejects unresolved robot semantics."""

import copy

import pytest
from test_doctor import write_json
from test_h1_pose_contract import body_contract, exported_manifest, full_stats, write_tiny_bundle
from zerith import artifact, model
from zerith.assets import sha256


def initializer(source, *, native=False):
    bundle = write_tiny_bundle(source)
    stats = full_stats(body_contract())
    if native:
        stats["command_contract"]["pose_frame"] = "source_native_shared_action_state_frame"
        stats["command_contract"]["waist_units"] = ["source_native"] * 3
    write_json(source / "config.json", bundle["config"])
    write_json(source / "norm.json", stats)
    manifest = dict(
        format="zerith-initializer-v1",
        config="config.json",
        normalization="norm.json",
        codec=dict(type="H1Command23Codec", state_dim=23, action_dim=23, embodiment_id=0),
        weights=dict(file="model.safetensors", sha256=bundle["weights_sha256"]),
    )
    write_json(source / "manifest.json", manifest)
    return bundle, stats


def test_source_native_initializer_fails_before_weights_or_new_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, stats = initializer(source, native=True)
    before = copy.deepcopy(stats)
    monkeypatch.setattr(artifact, "verified_file", lambda *args: pytest.fail("loaded weights"))
    with pytest.raises(ValueError, match="initializer norm.json.*H1 pose_frame"):
        artifact.prepare(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    from zerith.config import read_json

    assert read_json(source / "norm.json") == before


def test_verified_initializer_preserves_training_weights_and_loads_strictly(tmp_path):
    from safetensors.torch import load_file, save_file

    source = tmp_path / "source"
    source.mkdir()
    bundle, stats = initializer(source)
    path = source / "model.safetensors"
    state = load_file(str(path))
    state = {
        ("teacher_carrier." + k[len("world.") :] if k.startswith("world.") else k): v
        for k, v in state.items()
    }
    save_file(state, str(path))
    manifest = dict(
        format="zerith-initializer-v1",
        config="config.json",
        normalization="norm.json",
        codec=dict(type="H1Command23Codec", state_dim=23, action_dim=23, embodiment_id=0),
        weights=dict(file="model.safetensors", sha256=sha256(path)),
    )
    write_json(source / "manifest.json", manifest)
    exported = exported_manifest(stats, sha256(path))
    sidecar = tmp_path / "deployment.json"
    write_json(sidecar, {k: exported[k] for k in ["action_schema", "camera_schema", "vae_encoder"]})
    before = path.read_bytes()
    output = tmp_path / "output"
    prepared = artifact.prepare(source, output, deployment_metadata=sidecar)
    assert prepared["normalization_stats"] == stats
    assert prepared["weight_namespace"] == "training"
    assert (output / "model.safetensors").read_bytes() == before
    loaded, _ = model.load_policy(output, "cpu")
    assert set(loaded.state_dict()) == set(k.replace("teacher_carrier.", "world.") for k in state)
    assert bundle["normalization_stats"] == prepared["normalization_stats"]
