#!/usr/bin/env bash
# Download the complete asset set used by the LIBERO-Plus quick start.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZERITH_PYTHON="${ZERITH_PYTHON:-python}"

checkout_source() {
    local url="$1" destination="$2" revision="$3"
    if [[ ! -e "$destination" ]]; then
        git init -q "$destination"
        git -C "$destination" remote add origin "$url"
    elif [[ ! -d "$destination/.git" ]]; then
        echo "Not a Git checkout; leaving directory unchanged: $destination" >&2
        return 1
    fi
    if [[ "$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)" == "$revision" ]]; then
        echo "Reusing pinned source: $destination"
        return
    fi
    if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
        echo "Source has local changes at another revision; leaving it unchanged: $destination" >&2
        return 1
    fi
    if ! git -C "$destination" cat-file -e "$revision^{commit}" 2>/dev/null; then
        git -C "$destination" fetch --depth 1 origin "$revision"
    fi
    git -C "$destination" checkout --detach "$revision"
}

verify_files() {
    local path
    for path in "$@"; do
        if [[ ! -s "$path" ]]; then
            echo "Download incomplete: $path; check Hub connectivity/access and rerun." >&2
            return 1
        fi
    done
}

main() {
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        echo "Usage: bash scripts/download_libero_plus_assets.sh"
        echo "Run in zerith-inference; saved Hugging Face login or HF_TOKEN is reused."
        echo "Downloads the LIBERO-Plus policy, Wan/VGGT sources and weights, LIBERO"
        echo "sources, simulator assets and the 40 base instructions under simulation/assets."
        echo "Repeated runs reuse pinned Git checkouts and Hugging Face downloads."
        return
    fi
    if [[ $# -ne 0 ]]; then
        echo "Unknown argument; use --help." >&2
        return 2
    fi
    local program
    for program in git hf "$ZERITH_PYTHON"; do
        command -v "$program" >/dev/null || {
            echo "Missing $program; activate zerith-inference and install its dependencies." >&2
            return 1
        }
    done
    cd "$REPO_ROOT"
    "$ZERITH_PYTHON" -c 'from huggingface_hub import get_token' || return 1
    if ! "$ZERITH_PYTHON" -c 'from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)'; then
        hf auth login
    fi
    mkdir -p simulation/assets

    echo "[1/4] Downloading the released LIBERO-Plus policy"
    "$ZERITH_PYTHON" scripts/download_model.py --target libero-plus \
        --output simulation/assets/policies/libero-plus

    verify_files simulation/assets/policies/libero-plus/model.safetensors \
        simulation/assets/policies/libero-plus/manifest.json simulation/assets/policies/libero-plus/inference.json

    echo "[2/4] Downloading pinned encoder sources and weights"
    checkout_source https://github.com/Wan-Video/Wan2.2.git simulation/assets/Wan2.2 \
        42bf4cfaa384bc21833865abc2f9e6c0e67233dc
    checkout_source https://github.com/facebookresearch/vggt.git simulation/assets/vggt \
        a288dd0f14786c93483e45524328726ab7b1b4ce
    hf download Wan-AI/Wan2.2-TI2V-5B \
        --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
        --local-dir simulation/assets/Wan2.2-TI2V-5B \
        --include README.md Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth "google/umt5-xxl/*"
    verify_files simulation/assets/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
        simulation/assets/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
        simulation/assets/Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model \
        simulation/assets/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json
    hf download facebook/VGGT-1B \
        --revision 860abec7937da0a4c03c41d3c269c366e82abdf9 \
        --local-dir simulation/assets/VGGT-1B \
        --include README.md config.json model.safetensors

    verify_files simulation/assets/VGGT-1B/config.json simulation/assets/VGGT-1B/model.safetensors

    echo "[3/4] Downloading simulator sources and assets"
    checkout_source https://github.com/Lifelong-Robot-Learning/LIBERO.git simulation/assets/LIBERO \
        8f1084e3132a39270c3a13ebe37270a43ece2a01
    checkout_source https://github.com/sylvestf/LIBERO-plus.git simulation/assets/LIBERO-plus \
        4976dc30028e805ff8094b55501d532c48fec182
    hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset \
        --revision dd2bd61b7d9a6fef1abc52d606e983b41886a149 \
        --local-dir simulation/assets/libero-plus-download

    verify_files simulation/assets/libero-plus-download/assets.zip

    echo "[4/4] Exporting the 40 base instructions"
    "$ZERITH_PYTHON" scripts/export_libero_instructions.py \
        --libero-root simulation/assets/LIBERO --output simulation/assets/libero-instructions.json
    echo "Downloads complete. Continue with simulator environment configuration."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
