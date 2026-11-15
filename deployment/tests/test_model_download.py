"""Release downloads select the correct repository and preserve metadata."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("download_model", ROOT / "scripts/download_model.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


@pytest.mark.parametrize(
    "target,repo",
    [
        ("libero-plus", "OpenZerith/ZERITH-2.5-LIBERO-Plus"),
        ("libero", "OpenZerith/ZERITH-2.5-LIBERO"),
        ("metaworld", "OpenZerith/ZERITH-2.5-MetaWorld"),
        ("h1", "OpenZerith/ZERITH-2.5-H1"),
    ],
)
def test_download_uses_full_model_snapshot_and_selected_revision(
    monkeypatch, tmp_path, target, repo
):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    assert download.download_model(target, tmp_path, revision="release-v1") == tmp_path
    assert calls == [
        dict(repo_id=repo, repo_type="model", revision="release-v1", local_dir=str(tmp_path))
    ]


def test_list_needs_no_hub_dependency(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    download.main(["--list"])
    output = capsys.readouterr().out
    assert all(repo in output for repo in download.MODEL_REPOSITORIES.values())
    assert len(output.splitlines()) == 4


def test_unknown_target_is_rejected_before_download(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(SystemExit) as error:
        download.main(["--target", "robotwin", "--output", str(tmp_path)])
    assert error.value.code == 2
