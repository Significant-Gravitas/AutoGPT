"""Uploads scanned by a real ClamAV daemon: a private `clamav/clamav-debian`
container with a 2 MiB StreamMaxLength and only two synthetic signatures, one
in a file's body and one hashing a whole file. Gated on
`E2E_CLAMD_CONTAINER=1` + `docker` on PATH, marked `pytest.mark.slow`."""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from uuid import uuid4

import pytest

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.util import virus_scanner
from backend.util.virus_scanner import (
    VirusScannerService,
    VirusScannerSettings,
    scan_content_safe,
)

IMAGE = "clamav/clamav-debian:latest"
CONF_DIR = "/etc/clamav-test"
STREAM_LIMIT = 2 * 1024 * 1024
WITHIN_LIMIT = STREAM_LIMIT // 2
# Each half of this fits under the limit, and none is a whole file listed in
# the hash signatures.
OVER_LIMIT = 3 * STREAM_LIMIT // 2
BODY_SIGNATURE = "Test.Synthetic.Body"
HASH_SIGNATURE = "Test.Synthetic.WholeFile"
MARKER = b"synthetic-test-signature-for-whole-stream-scans"


def _straddling(size: int) -> bytes:
    """`size` zero bytes with `MARKER` across the middle, so scanning the file
    in halves would cut it in two."""
    start = size // 2 - len(MARKER) // 2
    return bytes(start) + MARKER + bytes(size - start - len(MARKER))


def _hashed(size: int) -> bytes:
    """`size` bytes the container's hash signatures list whole."""
    pattern = b"whole-file-hash-test."
    return (pattern * (size // len(pattern) + 1))[:size]


def _clamd_enabled() -> bool:
    return os.getenv("E2E_CLAMD_CONTAINER", "").lower() in ("1", "true", "yes")


clamd_container_only = pytest.mark.skipif(
    not (shutil.which("docker") and _clamd_enabled()),
    reason="real ClamAV e2e: requires docker + E2E_CLAMD_CONTAINER=1",
)


@clamd_container_only
@pytest.mark.slow
class TestScanningThroughRealClamd:
    @pytest.fixture(autouse=True)
    def scanner(self, clamd_port: int, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = VirusScannerSettings(
            clamav_service_host="127.0.0.1",
            clamav_service_port=clamd_port,
            # Bounds the one case where a refused stream's close is slow to
            # surface (a whole scan here is milliseconds).
            clamav_service_timeout=30,
        )
        monkeypatch.setattr(virus_scanner, "_scanner", VirusScannerService(settings))

    @pytest.mark.asyncio
    async def test_a_clean_file_within_the_limit_passes(self):
        await scan_content_safe(bytes(WITHIN_LIMIT), filename="clean.bin")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("make", "signature"),
        [(_straddling, BODY_SIGNATURE), (_hashed, HASH_SIGNATURE)],
        ids=["body-signature", "whole-file-hash"],
    )
    async def test_a_file_within_the_limit_is_scanned_whole(
        self, make: Callable[[int], bytes], signature: str
    ):
        with pytest.raises(VirusDetectedError) as caught:
            await scan_content_safe(make(WITHIN_LIMIT), filename="within.bin")

        assert caught.value.threat_name.startswith(signature)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "make",
        [_straddling, _hashed],
        ids=["body-signature-across-the-middle", "whole-file-hash"],
    )
    async def test_a_file_over_the_limit_is_refused_not_passed_clean(
        self, make: Callable[[int], bytes]
    ):
        # The daemon may name the size or just close the connection; the unit
        # tests pin each message. Here what matters is that it isn't clean.
        with pytest.raises(VirusScanError):
            await scan_content_safe(make(OVER_LIMIT), filename="over.bin")


@pytest.fixture(scope="module")
def clamd_port(tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    conf = tmp_path_factory.mktemp("clamd")
    (conf / "clamd.conf").write_bytes(
        b"TCPSocket 3310\n"
        b"Foreground yes\n"
        + f"DatabaseDirectory {CONF_DIR}\n".encode()
        + b"StreamMaxLength 2M\nMaxFileSize 2M\nMaxScanSize 2M\n"
    )
    (conf / "synthetic.ndb").write_bytes(
        f"{BODY_SIGNATURE}:0:*:{MARKER.hex()}\n".encode()
    )
    (conf / "synthetic.hdb").write_bytes(
        "".join(
            f"{hashlib.md5(data, usedforsecurity=False).hexdigest()}:{len(data)}:"
            f"{HASH_SIGNATURE}\n"
            for data in (_hashed(WITHIN_LIMIT), _hashed(OVER_LIMIT))
        ).encode()
    )
    name = f"virus-scanner-e2e-{uuid4().hex[:8]}"
    _docker(
        "create",
        "--name",
        name,
        "--memory",
        "3g",
        "--memory-swap",
        "3g",
        "-p",
        "127.0.0.1::3310",
        "-e",
        "CLAMAV_NO_FRESHCLAMD=true",
        IMAGE,
        # Options make the image's entrypoint run clamd alone, on this config.
        "--foreground",
        f"--config-file={CONF_DIR}/clamd.conf",
    )
    try:
        _docker("cp", f"{conf}{os.sep}.", f"{name}:{CONF_DIR}")
        _docker("start", name)
        port = int(_docker("port", name, "3310/tcp").splitlines()[0].split(":")[-1])
        _wait_for_pong(port, name)
        yield port
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name], capture_output=True, timeout=60, check=False
        )


def _docker(*args: str) -> str:
    done = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=120, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {done.stderr}")
    return done.stdout.strip()


def _wait_for_pong(port: int, name: str, deadline_s: float = 60.0) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as sock:
                sock.sendall(b"nPING\n")
                if sock.recv(16) == b"PONG\n":
                    return
        except OSError:
            pass
        time.sleep(0.5)
    logs = subprocess.run(
        ["docker", "logs", "--tail", "50", name],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    raise TimeoutError(
        f"clamd never answered PING within {deadline_s:.0f}s\n{logs.stdout}{logs.stderr}"
    )
