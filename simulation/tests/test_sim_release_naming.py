"""Renamed model loaders must keep existing release weights byte-compatible."""

import hashlib
import json

import pytest
import torch

from zerith_sim.models import ZerithConfig, ZerithPolicy
from zerith_sim.models.zerith import export_deployment_artifact, load_deployment_artifact


@pytest.mark.parametrize("format_name,architecture", [
    ("zerith-va-policy-action20-linear-g4d-teacher", "zerith-va-action20-linear-g4d-teacher"),
    ("fam-wam-policy-v5.2-action20-linear-g4d-teacher", "fam-wam-action20-linear-g4d-teacher-v5.2"),
])
def test_previous_release_names_load_without_rewriting_weights(tmp_path, format_name, architecture):
    policy = ZerithPolicy(ZerithConfig.tiny())
    manifest = export_deployment_artifact(
        policy, tmp_path, action_schema={"benchmark": "libero"},
        camera_schema={}, normalization_stats={}, vae_encoder_provenance={},
    )
    path = tmp_path / "model.safetensors"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["format"] = format_name
    manifest["config"]["architecture_version"] = architecture
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded, _ = load_deployment_artifact(tmp_path, verify_weight_sha256=True)
    assert loaded.config.architecture_version == "zerith-action20-linear-g4d-teacher"
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
