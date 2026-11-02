import asyncio
import sys
from pathlib import Path
from test.native_mobile.auth_process import auth_harness

import pytest


@pytest.mark.parametrize("phase", ["before", "after"])
async def test_noisy_harness_does_not_block_before_or_after_readiness(
    tmp_path: Path, phase: str
) -> None:
    completed = tmp_path / "finished"
    command = [
        sys.executable,
        "-c",
        r"""
import os, pathlib, sys, time
def noise():
    os.write(2, b"verbose diagnostic " * 65536 + b"\n")
if sys.argv[1] == "before": noise()
print("NATIVE_AUTH_READY http://127.0.0.1:12345", flush=True)
if sys.argv[1] == "after": noise()
pathlib.Path(sys.argv[2]).touch()
time.sleep(60)
""",
        phase,
        str(completed),
    ]

    async def wait_for_output() -> None:
        while not completed.exists():
            await asyncio.sleep(0.01)

    async with auth_harness(command, readiness_timeout=3) as origin:
        assert origin == "http://127.0.0.1:12345"
        await asyncio.wait_for(wait_for_output(), 3)


async def test_startup_failure_includes_stderr() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('startup diagnostic'); sys.exit(2)",
    ]
    with pytest.raises(RuntimeError, match="startup diagnostic"):
        async with auth_harness(command, readiness_timeout=3):
            pytest.fail("A crashed harness must not report readiness")


async def test_readiness_timeout_includes_bounded_diagnostic_tail() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys, time; sys.stderr.write('x' * 32768 + 'last startup diagnostic'); sys.stderr.flush(); time.sleep(60)",
    ]
    with pytest.raises(RuntimeError, match="last startup diagnostic") as error:
        async with auth_harness(command, readiness_timeout=1):
            pytest.fail("A harness without a ready marker must not report readiness")
    assert "timed out" in str(error.value)
    assert len(str(error.value)) < 8300
