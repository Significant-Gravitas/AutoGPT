#!/usr/bin/env bash
set -e
pip install -r requirements.txt
pip install pillow pytesseract
python -m pip install pywin32-postinstall-script --install
python - <<'PY'
import numpy, httpx
from packaging import version
if version.parse(numpy.__version__) >= version.parse("2.0"):
    print("WARNING: NumPy >= 2 not supported")
if version.parse(httpx.__version__) >= version.parse("0.28"):
    print("WARNING: httpx >= 0.28 not supported")
PY

echo "[*] Installing development tools..."
pip install -q \
  pytest \
  pytest-cov \
  black \
  pre-commit \
  ruff \
  mypy
