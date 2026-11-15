"""Asset download restarts must preserve edits in existing source checkouts."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/download_libero_plus_assets.sh"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def upstream(tmp_path):
    root = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    revisions = []
    for value in ("first", "second"):
        (root / "content.txt").write_text(value)
        git(root, "add", "content.txt")
        git(root, "-c", "user.name=Test", "-c", "user.email=unit-test", "commit", "-qm", value)
        revisions.append(git(root, "rev-parse", "HEAD"))
    return root, revisions


def checkout(source, destination, revision):
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; checkout_source "$2" "$3" "$4"',
            "--",
            str(SCRIPT),
            str(source),
            str(destination),
            revision,
        ],
        capture_output=True,
        text=True,
    )


def test_fresh_download_and_restart_preserve_local_changes(tmp_path, upstream):
    source, revisions = upstream
    target = tmp_path / "target"
    result = checkout(source, target, revisions[0])
    assert result.returncode == 0, result.stderr
    assert git(target, "rev-parse", "HEAD") == revisions[0]
    (target / "content.txt").write_text("local simulator configuration")
    result = checkout(source, target, revisions[0])
    assert result.returncode == 0, result.stderr
    assert (target / "content.txt").read_text() == "local simulator configuration"


def test_different_revision_with_local_changes_is_not_overwritten(tmp_path, upstream):
    source, revisions = upstream
    target = tmp_path / "target"
    assert checkout(source, target, revisions[1]).returncode == 0
    (target / "content.txt").write_text("keep my edits")
    result = checkout(source, target, revisions[0])
    assert result.returncode != 0
    assert git(target, "rev-parse", "HEAD") == revisions[1]
    assert (target / "content.txt").read_text() == "keep my edits"


def test_non_git_destination_is_not_modified(tmp_path, upstream):
    source, revisions = upstream
    target = tmp_path / "target"
    target.mkdir()
    (target / "notes.txt").write_text("keep")
    assert checkout(source, target, revisions[0]).returncode != 0
    assert (target / "notes.txt").read_text() == "keep"
