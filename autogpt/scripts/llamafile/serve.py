#!/usr/bin/env python3
"""
Use llamafile to serve a (quantized) mistral-7b-instruct-v0.2 model

Usage:
  cd <repo-root>/autogpt
  ./scripts/llamafile/serve.py
"""

import os
import platform
import subprocess
from pathlib import Path

LLAMAFILE = Path(
    "mistral-7b-instruct-v0.2.Q5_K_M.llamafile"
    + (".exe" if platform.system() == "Windows" else "")
)


def report_download_progress(chunk_number: int, chunk_size: int, total_size: int):
    if total_size != -1:
        downloaded_size = chunk_number * chunk_size
        percent = min(1, downloaded_size / total_size)
        bar = "#" * int(40 * percent)
        print(
            f"\rDownloading: [{bar:<40}] {percent:.0%}"
            f" - {downloaded_size/1e6:.1f}/{total_size/1e6:.1f} MB",
            end="",
        )


def download_llamafile():
    print(f"Downloading {LLAMAFILE.name}...")
    import urllib.request

    url = "https://huggingface.co/jartine/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.llamafile"  # noqa

    urllib.request.urlretrieve(url, LLAMAFILE.name, reporthook=report_download_progress)
    print()

    LLAMAFILE.chmod(0o755)
    subprocess.run([LLAMAFILE, "--version"], check=True)

    print(
        "\n"
        "NOTE: To use other models besides mistral-7b-instruct-v0.2, "
        "download them into autogpt/scripts/llamafile/"
    )


# Go to autogpt/scripts/llamafile/
os.chdir(Path(__file__).resolve().parent)

if not LLAMAFILE.is_file():
    download_llamafile()

subprocess.run(
    [LLAMAFILE, "--server", "--nobrowser", "--ctx-size", "0", "--n-predict", "1024"],
    check=True,
)

# note: --ctx-size 0 means the prompt context size will be set directly from the
# underlying model configuration. This may cause slow response times or consume
# a lot of memory.
