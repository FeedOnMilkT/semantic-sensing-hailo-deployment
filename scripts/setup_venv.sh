#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HAILO_VENV_DIR:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"
HAILO_WHEEL_DIR="${HAILO_WHEEL_DIR:-}"
HAILO_SITE_PACKAGES="${HAILO_SITE_PACKAGES:-}"

"$PYTHON" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install numpy onnx onnxruntime pyyaml

if [[ -n "$HAILO_WHEEL_DIR" ]]; then
  if [[ ! -d "$HAILO_WHEEL_DIR" ]]; then
    echo "HAILO_WHEEL_DIR does not exist: $HAILO_WHEEL_DIR" >&2
    exit 2
  fi
  "$VENV_PYTHON" -m pip install --no-index --find-links "$HAILO_WHEEL_DIR" hailo-dataflow-compiler
fi

if [[ -n "$HAILO_SITE_PACKAGES" ]]; then
  if [[ ! -d "$HAILO_SITE_PACKAGES" ]]; then
    echo "HAILO_SITE_PACKAGES does not exist: $HAILO_SITE_PACKAGES" >&2
    exit 2
  fi
  SITE_FILE="$VENV_DIR/lib/$($VENV_PYTHON -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/hailo-sdk.pth"
  printf '%s\n' "$HAILO_SITE_PACKAGES" > "$SITE_FILE"
fi

if ! "$VENV_PYTHON" "$ROOT/scripts/check_hailo_env.py"; then
  echo "\nThe venv is ready for host-side tools, but Hailo SDK commands need the authorized DFC 3.33.x installation."
fi
printf '\nEnvironment created at %s\nActivate with: source %s/bin/activate\n' "$VENV_DIR" "$VENV_DIR"
