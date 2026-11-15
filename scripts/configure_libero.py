"""Configure an isolated LIBERO checkout and unpack verified LIBERO-Plus assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

PLUS_ASSET_SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"


def extract_assets(archive: Path, package: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("LIBERO-Plus asset archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as source:
        members = []
        for member in source.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe asset archive path")
            if "assets" not in relative.parts or member.is_dir():
                continue
            suffix = relative.parts[relative.parts.index("assets") :]
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Asset archive symlinks are unsupported")
            members.append((member, package.joinpath(*suffix)))
        if not members:
            raise ValueError("Archive contains no assets directory")
        for member, destination in members:
            if destination.is_symlink() or any(
                parent.is_symlink() for parent in destination.parents
            ):
                raise ValueError("Asset destination contains a symlink")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def configure(benchmark_root: Path, config_dir: Path, asset_archive: Path | None = None) -> None:
    benchmark_root = benchmark_root.resolve()
    package = benchmark_root / "libero/libero"
    if not (package / "benchmark/__init__.py").is_file():
        raise ValueError("Expected an upstream LIBERO or LIBERO-Plus checkout")
    if asset_archive is not None:
        extract_assets(asset_archive, package, PLUS_ASSET_SHA256)
    if not (package / "assets").is_dir():
        raise ValueError("Simulator assets are missing")
    # Both upstream repositories omit this marker, which editable discovery needs.
    marker = benchmark_root / "libero/__init__.py"
    if not marker.exists():
        marker.write_text('"""LIBERO namespace package."""\n')
    wrapper = package / "envs/env_wrapper.py"
    if wrapper.is_file():
        # Make ImageMagick return encoded PNG bytes rather than a raw pixel buffer.
        text = wrapper.read_text()
        updated = text.replace("x.make_blob()", 'x.make_blob(format="PNG")')
        if updated != text:
            wrapper.write_text(updated)
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(
        benchmark_root=str(package),
        bddl_files=str(package / "bddl_files"),
        init_states=str(package / "init_files"),
        assets=str(package / "assets"),
        datasets=str(package.parent / "datasets"),
    )
    # JSON is valid YAML and avoids importing simulator dependencies here.
    (config_dir / "config.yaml").write_text(json.dumps(paths, indent=2) + "\n")
    print("LIBERO source, assets and isolated configuration are ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--asset-archive", type=Path)
    args = parser.parse_args()
    configure(args.benchmark_root, args.config_dir, args.asset_archive)


if __name__ == "__main__":
    main()
