#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${ZERITH_PYTHON:-python}" -m zerith.process stop h1 "$@"
