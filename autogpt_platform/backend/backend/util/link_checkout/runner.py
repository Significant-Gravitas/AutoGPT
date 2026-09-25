"""Spawns the payment worker for one job.

The worker runs as a fresh interpreter in isolated mode (``-I``: no user site
packages, no ``PYTHON*`` variables), with only ``PATH`` and the egress proxy in
its environment, in its own session so a timeout can kill everything it
started. Its stderr is discarded and its stdout is bounded and parsed as one
``WorkerResult``; a failure reports nothing more specific than "failed".
"""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

from backend.util.link_checkout.config import https_proxy
from backend.util.link_checkout.models import WorkerJob, WorkerResult

_ROOT = str(Path(__file__).resolve().parents[3])
_ENTRY = (
    "import sys,logging; logging.disable(logging.CRITICAL); "
    f"sys.path.insert(0, {_ROOT!r}); "
    "from backend.util.link_checkout.worker import main; main()"
)


async def run_worker(job: WorkerJob) -> WorkerResult:
    payload = job.model_dump(mode="json")
    payload["access_token"] = job.access_token.get_secret_value()
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if proxy := https_proxy():
        environment["CHECKOUT_HTTPS_PROXY"] = proxy
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-B",
        "-c",
        _ENTRY,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
        env=environment,
    )
    try:
        output, _ = await asyncio.wait_for(
            process.communicate(json.dumps(payload).encode()), 65
        )
        if process.returncode or len(output) > 8192:
            raise RuntimeError("Private checkout worker failed")
        return WorkerResult.model_validate_json(output)
    except BaseException:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        raise RuntimeError(
            "Private checkout worker failed; never retry payment"
        ) from None
