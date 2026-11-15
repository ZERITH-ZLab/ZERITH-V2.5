"""Download one supported ZERITH 2.5 model release from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_REPOSITORIES = {
    "libero-plus": "OpenZerith/ZERITH-2.5-LIBERO-Plus",
    "libero": "OpenZerith/ZERITH-2.5-LIBERO",
    "metaworld": "OpenZerith/ZERITH-2.5-MetaWorld",
    "h1": "OpenZerith/ZERITH-2.5-H1",
}


# Release revisions include ZERITH 2.5 naming, license text and attribution.
MODEL_REVISIONS = {
    "libero-plus": "63b93338e419e122c1b22ae7bbd20bbf9adc534e",
    "libero": "968786e834815ef7305afadd93533d7cb83fa997",
    "metaworld": "85c96232e0119fb6a4081ea2bc2e6ab71ef33cae",
    "h1": "cb189887e12825ddde2941f2b7a91de5fe984040",
}


def download_model(target: str, output: str | Path, revision: str | None = None) -> Path:
    repo = MODEL_REPOSITORIES[target]
    from huggingface_hub import snapshot_download

    # Use the saved Hub login or HF_TOKEN; download manifests and terms as well.
    return Path(
        snapshot_download(
            repo_id=repo,
            repo_type="model",
            revision=revision or MODEL_REVISIONS[target],
            local_dir=str(Path(output).expanduser().resolve()),
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=MODEL_REPOSITORIES)
    parser.add_argument("--output", help="directory for the complete model snapshot")
    parser.add_argument(
        "--revision", help="override the pinned Hub commit with a commit, tag or branch"
    )
    parser.add_argument("--list", action="store_true", help="list supported model repositories")
    args = parser.parse_args(argv)
    if args.list:
        for target, repo in MODEL_REPOSITORIES.items():
            print(f"{target}\t{repo}")
        return
    if args.target is None or args.output is None:
        parser.error("--target and --output are required unless --list is used")
    print(download_model(args.target, args.output, args.revision))


if __name__ == "__main__":
    main()
