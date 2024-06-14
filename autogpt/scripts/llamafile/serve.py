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

LLAMAFILE = Path("mistral-7b-instruct-v0.2.Q5_K_M.llamafile")
LLAMAFILE_URL = f"https://huggingface.co/jartine/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/{LLAMAFILE.name}"  # noqa
LLAMAFILE_EXE = Path("llamafile.exe")
LLAMAFILE_EXE_URL = "https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.6/llamafile-0.8.6"  # noqa


def download_file(url: str, to_file: Path) -> None:
    print(f"Downloading {to_file.name}...")
    import urllib.request

    urllib.request.urlretrieve(url, to_file, reporthook=report_download_progress)
    print()


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


if __name__ == "__main__":
    # Go to autogpt/scripts/llamafile/
    os.chdir(Path(__file__).resolve().parent)

    if not LLAMAFILE.is_file():
        download_file(LLAMAFILE_URL, LLAMAFILE)

        if platform.system() != "Windows":
            LLAMAFILE.chmod(0o755)
            subprocess.run([LLAMAFILE, "--version"], check=True)

        print(
            "\n"
            "NOTE: To use other models besides mistral-7b-instruct-v0.2, "
            "download them into autogpt/scripts/llamafile/"
        )

    if platform.system() != "Windows":
        base_command = [LLAMAFILE]
    else:
        # Windows does not allow executables over 4GB, so we have to download a
        # model-less llamafile.exe and extract the model weights (.gguf file)
        # from the downloaded .llamafile.
        if not LLAMAFILE_EXE.is_file():
            download_file(LLAMAFILE_EXE_URL, LLAMAFILE_EXE)
            LLAMAFILE_EXE.chmod(0o755)
            subprocess.run([LLAMAFILE_EXE, "--version"], check=True)

        model_file = LLAMAFILE.with_suffix(".gguf")
        if not model_file.is_file():
            import zipfile

            with zipfile.ZipFile(LLAMAFILE, "r") as zip_ref:
                gguf_file = next(
                    (file for file in zip_ref.namelist() if file.endswith(".gguf")),
                    None,
                )
                if not gguf_file:
                    raise Exception("No .gguf file found in the zip file.")

                zip_ref.extract(gguf_file)
                Path(gguf_file).rename(model_file)

        base_command = [LLAMAFILE_EXE, "-m", model_file]

    subprocess.run(
        [
            *base_command,
            "--server",
            "--nobrowser",
            "--ctx-size",
            "0",
            "--n-predict",
            "1024",
        ],
        check=True,
    )

    # note: --ctx-size 0 means the prompt context size will be set directly from the
    # underlying model configuration. This may cause slow response times or consume
    # a lot of memory.
