#!/usr/bin/env bash
# Install LIBERO host libraries without replacing container-mounted GPU libraries.
set -euo pipefail

if [[ "$EUID" -ne 0 ]]; then
    echo "Run with sudo, or as root." >&2
    exit 1
fi

packages=(git curl libmagickwand-dev libexpat1 libfontconfig1 libglvnd0 libglx0)
for library in EGL GL; do
    if ! python3 - "$library" <<'PYTHON'
import ctypes.util
import os
from pathlib import Path
import sys
name = sys.argv[1]
folders = os.environ.get('LD_LIBRARY_PATH', '').split(':') + [
    '/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu', '/usr/local/nvidia/lib64'
]
assert ctypes.util.find_library(name) or any(
    (Path(folder) / ('lib' + name + '.so.1')).is_file() for folder in folders if folder
)
PYTHON
    then
        case "$library" in
            EGL) packages+=(libegl1) ;;
            GL) packages+=(libgl1) ;;
        esac
    fi
done

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"

python3 -c 'import ctypes; ctypes.CDLL("libEGL.so.1"); ctypes.CDLL("libGL.so.1")'
