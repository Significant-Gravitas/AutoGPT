#!/usr/bin/env python3
"""Print the APT_REFRESH build argument for the image builds.

The backend and single-container Dockerfiles declare APT_REFRESH in front of
each stage's package installs. It is part of the cache key of every RUN after
it, so a new value re-runs the installs and an unchanged one lets BuildKit
reuse the cached layers.

The value is the UTC date plus a short hash of the package indexes in Debian's
security archive, so it changes once a day and again whenever Debian publishes
security updates. With the date alone, an update published during the day
stayed behind that day's cached layer until midnight, and once the
single-container image's vulnerability scan learnt of the fix, every build
failed it until then. Only the package index lines are hashed, because the
archive re-signs the file, with a new Date, without publishing anything.

If the index can't be read after a few tries, the value is the date alone. A
build that falls back misses the cache and re-runs the package installs, a few
minutes, though with current packages. On a dev push it also writes its layers
to the shared registry cache in place of the hashed key's, so the next build
that reads the index re-runs them as well. Failed reads are rare and only cost
time, so the fallback keeps no state to reuse the last key: the
single-container workflow keeps nothing in the Actions cache.
"""

from __future__ import annotations

import hashlib
import http.client
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Callable

# The Debian release of the images' package-installing stages (debian:13-slim).
# test_dockerfile_layers.py fails when the Dockerfiles move to another one.
CODENAME = "trixie"
SUITE = f"{CODENAME}-security"
# The host the base image's apt reads the security archive from, so the key
# never runs ahead of the index the package installs see.
INDEX_URL = f"https://deb.debian.org/debian-security/dists/{SUITE}/InRelease"
# The images install from the main component, for whichever architecture.
PACKAGE_INDEX = re.compile(r"main/binary-[^/]+/Packages(?:\.\w+)?")
ATTEMPTS = 3
TIMEOUT_SECONDS = 10
FIRST_RETRY_DELAY_SECONDS = 2


def refresh_key(
    fetch: Callable[[str], str] | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """The UTC date plus a hash of the security archive's package indexes."""
    date = (now or utc_now)().astimezone(timezone.utc).date().isoformat()
    failure = ""
    for attempt in range(ATTEMPTS):
        if attempt:
            sleep(FIRST_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
        try:
            release = (fetch or read_index)(INDEX_URL)
            key = f"{date}+{index_digest(release)}"
        except (OSError, ValueError, http.client.HTTPException) as error:
            failure = " ".join(str(error).split()) or type(error).__name__
            continue
        published = re.search(r"^Date: *(.+?) *$", release, re.M)
        print(
            f"APT_REFRESH={key} from {INDEX_URL}"
            f" ({published[1] if published else 'undated'})",
            file=sys.stderr,
        )
        return key
    print(
        f"::warning title=APT_REFRESH is the date alone::Could not read"
        f" {INDEX_URL} ({failure}), so this build re-runs the package installs.",
        file=sys.stderr,
    )
    return date


def index_digest(release: str) -> str:
    """Hash the package index lines of a Release file's SHA256 list."""
    codename = re.search(r"^Codename: *(\S+) *$", release, re.M)
    if codename is None or codename[1] != SUITE:
        raise ValueError(f"the file read is not the {SUITE} Release file")
    checksums = re.search(r"^SHA256:\n((?:[ \t].*(?:\n|$))*)", release, re.M)
    indexes = [
        line.strip()
        for line in (checksums[1] if checksums else "").splitlines()
        if len(fields := line.split()) == 3 and PACKAGE_INDEX.fullmatch(fields[2])
    ]
    if not indexes:
        raise ValueError(f"the {SUITE} Release file lists no package indexes")
    return hashlib.sha256("\n".join(indexes).encode()).hexdigest()[:12]


def read_index(url: str) -> str:
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print(refresh_key())
