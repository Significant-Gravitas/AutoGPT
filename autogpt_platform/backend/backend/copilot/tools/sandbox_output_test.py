"""What ``run_sandboxed`` keeps of a command's output, and when it stops reading.

bwrap is swapped for the bare command, so this runs where bwrap cannot."""

import asyncio
from unittest.mock import patch

import pytest

from .sandbox import MAX_OUTPUT_SIZE, get_workspace_dir, run_sandboxed

_NOTE = f"Printed more than {MAX_OUTPUT_SIZE:,} bytes and was killed"


@pytest.fixture(autouse=True)
def without_bwrap():
    with (
        patch("backend.copilot.tools.sandbox.has_full_sandbox", return_value=True),
        patch(
            "backend.copilot.tools.sandbox._build_bwrap_command",
            side_effect=lambda command, cwd, env: command,
        ),
    ):
        yield


async def _run(script: str, timeout: int) -> tuple[str, str, int, bool]:
    cwd = get_workspace_dir("sandbox-output-test")
    return await asyncio.wait_for(
        run_sandboxed(["bash", "-c", script], cwd=cwd, timeout=timeout), timeout=30
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
async def test_command_that_keeps_printing_is_killed_at_the_cap(stream: str):
    stdout, stderr, exit_code, timed_out = await _run(f"yes >/dev/{stream}", 20)

    kept = "y\n" * (MAX_OUTPUT_SIZE // 2)
    if stream == "stdout":
        assert (stdout, stderr) == (kept, _NOTE)
    else:
        assert (stdout, stderr) == ("", f"{_NOTE}\n{kept}".rstrip("\n"))
    assert (exit_code, timed_out) == (-1, False)


@pytest.mark.asyncio
async def test_output_up_to_the_cap_is_returned_whole():
    stdout, stderr, exit_code, timed_out = await _run(
        f"yes | head -c {MAX_OUTPUT_SIZE}; echo done >&2; exit 3", 20
    )

    assert stdout == "y\n" * (MAX_OUTPUT_SIZE // 2)
    assert (stderr, exit_code, timed_out) == ("done\n", 3, False)


@pytest.mark.asyncio
async def test_a_process_the_kill_missed_cannot_hold_the_run_open():
    # A process outside the stopped group must not keep the run open.
    script = "setsid bash -c 'sleep 2; while :; do echo y; sleep 0.01; done' & sleep 60"

    stdout, stderr, exit_code, timed_out = await _run(script, 1)

    assert (stdout, stderr) == ("", "Execution timed out after 1s")
    assert (exit_code, timed_out) == (-1, True)
