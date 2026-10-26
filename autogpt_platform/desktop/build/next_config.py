"""Stop the Next standalone server writing into the bundle.

A standalone build carries its configuration inside `server.js` (and again in
`.next/required-server-files.json`). As built, the server keeps three caches
on disk under `.next/cache`, next to its own code: optimised images, `fetch`
responses, and regenerated pages. Installed, that directory is read-only (a
.deb, an AppImage) or sealed by a code signature (macOS), and what was cached
would not survive an upgrade anyway.

`experimental.isrFlushToDisk = false` turns all three off; pages built ahead
of time are still read from `.next/server`. `images.maximumDiskCacheSize = 0`
turns the image cache off by itself as well, in case a later Next drops the
first key. Without a disk cache every image is fetched and encoded again once
the browser's copy is older than `images.minimumCacheTTL`, which is a minute
as built: four hours keeps a working session inside the browser's own cache.
The price is that a remote image replaced under the same URL can be that
stale.

The frontend's source and build are untouched; this edits the build's output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EMBEDDED_CONFIG = re.compile(r"^const nextConfig = (\{.*\})$", re.MULTILINE)
IMAGE_CACHE_TTL_SECONDS = 4 * 60 * 60
UPDATE = (
    "Read how this version of Next decides where its caches go "
    "(next/dist/server/image-optimizer.js, lib/incremental-cache/file-system-cache.js) "
    "and update desktop/build/next_config.py."
)


class NextConfigError(RuntimeError):
    pass


def keep_caches_off_disk(frontend: Path) -> None:
    """Rewrite both copies of the configuration in an assembled frontend."""
    server = frontend / "server.js"
    _write(server, rewrite_server_js(_read(server)))
    required = frontend / ".next" / "required-server-files.json"
    _write(required, rewrite_required_files(_read(required)))
    check(frontend)


def check(frontend: Path) -> None:
    """Raise unless both copies have the caches off."""
    embedded = EMBEDDED_CONFIG.findall(_read(frontend / "server.js"))
    if len(embedded) != 1:
        raise NextConfigError(f"server.js: {_shape_problem(len(embedded))}")
    required = json.loads(_read(frontend / ".next" / "required-server-files.json"))
    copies = {
        "server.js": json.loads(embedded[0]),
        "required-server-files.json": required["config"],
    }
    for name, config in copies.items():
        wrong = [
            key for key, value in _settings(config).items() if value != WANTED[key]
        ]
        if wrong:
            raise NextConfigError(
                f"{name} still has the as-built value of {', '.join(wrong)}"
            )


def rewrite_server_js(text: str) -> str:
    matches = list(EMBEDDED_CONFIG.finditer(text))
    if len(matches) != 1:
        raise NextConfigError(f"server.js: {_shape_problem(len(matches))}")
    start, end = matches[0].span(1)
    config = adjust(json.loads(matches[0].group(1)), "server.js")
    # Spliced in by position: the configuration holds backslashes (a regular
    # expression, Windows paths) that a regex replacement would read as escapes.
    return (
        text[:start]
        + json.dumps(config, separators=(",", ":"), ensure_ascii=False)
        + text[end:]
    )


def rewrite_required_files(text: str) -> str:
    document = json.loads(text)
    if not isinstance(document.get("config"), dict):
        raise NextConfigError(
            f"required-server-files.json has no `config` object. {UPDATE}"
        )
    document["config"] = adjust(document["config"], "required-server-files.json")
    return json.dumps(document, indent=2, ensure_ascii=False)


def adjust(config: dict, where: str) -> dict:
    experimental, images = config.get("experimental"), config.get("images")
    if not isinstance(experimental, dict) or "isrFlushToDisk" not in experimental:
        # Setting a key Next no longer reads would pass here and write to
        # disk again in the installed app.
        raise NextConfigError(
            f"{where}: the configuration has no experimental.isrFlushToDisk, the switch "
            f"that keeps Next's caches off the disk. {UPDATE}"
        )
    if not isinstance(images, dict) or "minimumCacheTTL" not in images:
        raise NextConfigError(
            f"{where}: the configuration has no images.minimumCacheTTL. {UPDATE}"
        )
    experimental["isrFlushToDisk"] = WANTED["experimental.isrFlushToDisk"]
    images["maximumDiskCacheSize"] = WANTED["images.maximumDiskCacheSize"]
    images["minimumCacheTTL"] = WANTED["images.minimumCacheTTL"]
    return config


WANTED = {
    "experimental.isrFlushToDisk": False,
    "images.maximumDiskCacheSize": 0,
    "images.minimumCacheTTL": IMAGE_CACHE_TTL_SECONDS,
}


def _settings(config: dict) -> dict[str, object]:
    return {
        "experimental.isrFlushToDisk": config.get("experimental", {}).get(
            "isrFlushToDisk"
        ),
        "images.maximumDiskCacheSize": config.get("images", {}).get(
            "maximumDiskCacheSize"
        ),
        "images.minimumCacheTTL": config.get("images", {}).get("minimumCacheTTL"),
    }


def _shape_problem(found: int) -> str:
    return (
        f"expected one line `const nextConfig = {{...}}` holding the build's "
        f"configuration, found {found}. {UPDATE}"
    )


def _read(path: Path) -> str:
    # No newline translation either way: the pattern is anchored on line
    # ends, and the file must come back byte for byte but for the edit.
    with open(path, encoding="utf-8", newline="") as stream:
        return stream.read()


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as stream:
        stream.write(text)
