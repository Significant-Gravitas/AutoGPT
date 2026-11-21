"""The six public data files used by all seven tiktoken 0.14.0 encodings.

Pins come from tiktoken_ext.openai_public in the locked release. SHA-1 names
are tiktoken.load's URL cache keys; only SHA-256 verifies the contents.
"""

from __future__ import annotations

import hashlib
import shutil
import tomllib
from pathlib import Path

from artifacts import Artifact
from fetch import download

VERSION = "0.14.0"
CACHE_PATH = Path("assets") / "tiktoken"
ENCODINGS = (
    "gpt2",
    "r50k_base",
    "p50k_base",
    "p50k_edit",
    "cl100k_base",
    "o200k_base",
    "o200k_harmony",
)
ASSETS = (
    Artifact(
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        # pragma: allowlist nextline secret
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
    Artifact(
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
        # pragma: allowlist nextline secret
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    ),
    Artifact(
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        # pragma: allowlist nextline secret
        "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    ),
    Artifact(
        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
        # pragma: allowlist nextline secret
        "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
    ),
    Artifact(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        # pragma: allowlist nextline secret
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    ),
    Artifact(
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        # pragma: allowlist nextline secret
        "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    ),
)


def cache_key(artifact: Artifact) -> str:
    return hashlib.sha1(artifact.url.encode(), usedforsecurity=False).hexdigest()


def check_lock(lock: Path) -> None:
    with lock.open("rb") as source:
        packages = tomllib.load(source)["package"]
    versions = [p["version"] for p in packages if p["name"] == "tiktoken"]
    if versions != [VERSION]:
        raise RuntimeError(
            f"tokenizer assets require tiktoken {VERSION}, but the lock has {versions}; "
            "review the encoding constructors and pins in build/tokenizer_assets.py"
        )


def install(runtime: Path, cache: Path, lock: Path) -> None:
    check_lock(lock)
    destination = runtime / CACHE_PATH
    destination.mkdir(parents=True, exist_ok=True)
    for artifact in ASSETS:
        name = cache_key(artifact)
        source = download(artifact, cache / "tiktoken", name=name)
        shutil.copy2(source, destination / name)
    verify(destination)


def verify(cache: Path) -> None:
    for artifact in ASSETS:
        path = cache / cache_key(artifact)
        if not path.is_file():
            raise RuntimeError(f"missing tokenizer asset: {artifact.filename}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != artifact.sha256:
            raise RuntimeError(f"tokenizer asset sha256 mismatch: {artifact.filename}")
