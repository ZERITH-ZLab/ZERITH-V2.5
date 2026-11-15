"""Historic normalization hashes survive the public package rename."""

import copy
import json

import torch
from test_h1_pose_contract import body_contract, full_stats
from zerith_core.actions.h1 import H1Command23Codec, digest


def test_legacy_h1_statistics_remain_unmodified_and_keep_their_fingerprint():
    stats = copy.deepcopy(full_stats(body_contract()))
    stats["format"] = "zvla-h1-command23-minmax-v1"
    stats["arms_minmax"]["format"] = "zvla-mixed-action20-native7-state20-minmax-v1"
    stats["arms_minmax_sha256"] = digest(stats["arms_minmax"])
    before = json.dumps(stats, sort_keys=True)
    codec = H1Command23Codec(stats, embodiments=("h1",))
    assert json.dumps(stats, sort_keys=True) == before
    assert codec.stats_fingerprint == digest(stats)
    torch.testing.assert_close(
        codec.stats_sha256,
        torch.tensor(list(bytes.fromhex(digest(stats))), dtype=torch.uint8),
        rtol=0,
        atol=0,
    )


def test_legacy_camera_export_prepares_without_changing_weights_or_statistics(tmp_path):
    import hashlib

    from test_h1_pose_contract import exported_manifest
    from zerith import artifact
    from zerith.contract import CAMERAS, metadata, validate_metadata

    source, destination = tmp_path / "source", tmp_path / "bundle"
    source.mkdir()
    weights = b"unchanged fixture tensor bytes"
    (source / "model.safetensors").write_bytes(weights)
    stats = full_stats(body_contract())
    old = exported_manifest(stats, hashlib.sha256(weights).hexdigest())
    profile = old["camera_schema"]["profiles"].pop("zerith_raw_3canonical_view_v1")
    profile["source_camera_names"] = [f"fam_wam_raw/canonical_view_{i}" for i in range(3)]
    old["camera_schema"]["profiles"]["fam_wam_raw_3canonical_view_v1"] = profile
    (source / "manifest.json").write_text(json.dumps(old))
    result = artifact.prepare(source, destination)
    assert (destination / "model.safetensors").read_bytes() == weights
    assert result["normalization_stats"] == stats
    assert metadata()["camera_names"] == list(CAMERAS)
    legacy_health = metadata()
    legacy_health["camera_names"] = profile["source_camera_names"]
    validate_metadata(legacy_health)
    legacy_health["camera_names"] = list(reversed(profile["source_camera_names"]))
    import pytest

    with pytest.raises(ValueError, match="camera_names"):
        validate_metadata(legacy_health)
    profile["source_camera_names"].reverse()
    (source / "manifest.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="camera order"):
        artifact.prepare(source, tmp_path / "rejected")
