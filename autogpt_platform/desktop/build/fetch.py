"""Download and unpack build inputs, with a local cache and digest checks."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from artifacts import Artifact


def download(artifact: Artifact, cache: Path, name: str | None = None) -> Path:
    """Fetch into the cache (as `name`, when the URL's own file name is not
    unique enough) and verify the pinned digest."""
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (name or artifact.filename)
    if not target.exists():
        print(f"  downloading {artifact.url}", flush=True)
        partial = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(
            artifact.url, headers={"User-Agent": "autogpt-desktop-build"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, open(
            partial, "wb"
        ) as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        partial.replace(target)
    digest = _sha256(target)
    if digest != artifact.sha256:
        target.unlink()
        raise RuntimeError(
            f"{artifact.filename}: sha256 {digest} does not match the pinned {artifact.sha256}"
        )
    return target


def extract(archive: Path, destination: Path, *, strip_top_level: bool = False) -> None:
    """Unpack into `destination`. With `strip_top_level`, an archive whose
    content sits in a single top-level directory lands directly in
    `destination` (the equivalent of tar's --strip-components=1)."""
    if destination.exists():
        shutil.rmtree(destination)
    staging = destination.with_name(destination.name + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(staging, filter="tar")
    children = list(staging.iterdir())
    if strip_top_level and len(children) == 1 and children[0].is_dir():
        children[0].replace(destination)
        staging.rmdir()
    else:
        staging.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
