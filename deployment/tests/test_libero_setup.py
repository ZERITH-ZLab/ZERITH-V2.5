"""Simulator setup unpacks trusted assets without escaping its directory."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/configure_libero.py"
SPEC = importlib.util.spec_from_file_location("configure_libero", SCRIPT)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def archive_with(tmp_path, name, content=b"mesh"):
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, content)
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def test_nested_archive_assets_are_installed_at_package_root(tmp_path):
    archive, digest = archive_with(tmp_path, "upstream/release/assets/scenes/object.xml")
    package = (tmp_path / "package").resolve()
    setup.extract_assets(archive, package, digest)
    assert (package / "assets/scenes/object.xml").read_bytes() == b"mesh"
    assert not (package / "upstream").exists()


def test_unverified_archive_is_rejected_before_extraction(tmp_path):
    archive, _ = archive_with(tmp_path, "assets/scenes/object.xml")
    package = tmp_path / "package"
    with pytest.raises(ValueError, match="SHA256"):
        setup.extract_assets(archive, package, "0" * 64)
    assert not package.exists()


@pytest.mark.parametrize("name", ["../assets/object.xml", "/assets/object.xml"])
def test_archive_paths_cannot_escape_destination(tmp_path, name):
    archive, digest = archive_with(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe"):
        setup.extract_assets(archive, tmp_path / "package", digest)
    assert not (tmp_path / "package").exists()


def test_archive_symlinks_are_rejected(tmp_path):
    member = zipfile.ZipInfo("assets/link")
    member.create_system = 3
    member.external_attr = 0o120777 << 16
    archive, digest = archive_with(tmp_path, member, b"/outside")
    with pytest.raises(ValueError, match="symlinks"):
        setup.extract_assets(archive, tmp_path / "package", digest)


def test_configuration_is_isolated_and_repeatable(tmp_path):
    root = tmp_path / "LIBERO"
    package = root / "libero/libero"
    (package / "benchmark").mkdir(parents=True)
    (package / "benchmark/__init__.py").write_text("")
    (package / "assets").mkdir()
    (package / "envs").mkdir()
    wrapper = package / "envs/env_wrapper.py"
    wrapper.write_text("result = x.make_blob()\n")
    config = tmp_path / "config"
    setup.configure(root, config)
    first = (config / "config.yaml").read_bytes()
    setup.configure(root, config)
    assert (config / "config.yaml").read_bytes() == first
    assert json.loads(first)["assets"] == str(package / "assets")
    assert wrapper.read_text() == 'result = x.make_blob(format="PNG")\n'
    assert (root / "libero/__init__.py").is_file()
