"""Each bash_exec sandbox keeps its own processes and IPC objects."""

import asyncio
from pathlib import Path

import pytest

from backend.copilot.tools.sandbox import (
    PROC_MOUNT_REFUSED,
    _build_bwrap_command,
    explain_proc_mount_failure,
    has_full_sandbox,
    run_sandboxed,
)


def test_bwrap_gives_each_sandbox_its_own_processes_and_ipc():
    command = _build_bwrap_command(["true"], "/tmp/copilot-test", {"PATH": "/bin"})

    flags = command[: command.index("--")]

    assert "--unshare-pid" in flags
    assert "--unshare-ipc" in flags
    assert "--unshare-net" in flags


@pytest.mark.parametrize(
    "stderr",
    [
        # bwrap fills its own destination in, and which one depends on the version.
        "bwrap: Can't mount proc on /proc: Operation not permitted",
        "bwrap: Can't mount proc on /newroot/proc: Operation not permitted",
    ],
)
def test_a_refused_proc_mount_names_the_host_setting(stderr):
    explained = explain_proc_mount_failure(stderr)

    assert "systempaths=unconfined" in explained
    assert "procMount: Unmasked" in explained
    assert stderr in explained


def test_a_commands_own_stderr_is_left_alone():
    assert explain_proc_mount_failure("ls: /nope: No such file") == (
        "ls: /nope: No such file"
    )


# The test below runs two real sandboxes, and skips only where bubblewrap
# cannot create namespaces at all.


@pytest.mark.asyncio
async def test_sandbox_cannot_read_another_sandboxes_command_line(tmp_path):
    other = tmp_path / "other"
    scanner = tmp_path / "scanner"
    other.mkdir()
    scanner.mkdir()
    if not await _sandbox_runs_here(str(scanner)):
        pytest.skip("bubblewrap cannot create namespaces on this host")

    marker = "Bearer ghp_another_chats_token"
    other_chat = asyncio.create_task(
        run_sandboxed(
            # Holds its argv until released, so the scan runs while it is up.
            ["sh", "-c", "touch started; until [ -e go ]; do sleep 0.05; done"]
            + ["other-chat", marker],
            str(other),
        )
    )
    try:
        await _until_exists(other / "started", other_chat)
        stdout, _, exit_code, _ = await run_sandboxed(
            [
                "sh",
                "-c",
                'cat /proc/[0-9]*/cmdline 2>/dev/null | tr "\\0" " "',
            ],
            str(scanner),
        )
        assert not other_chat.done(), "the other chat ended before the scan finished"
    finally:
        (other / "go").touch()
        _, _, other_exit_code, other_timed_out = await other_chat

    assert exit_code == 0
    # Without its own command line in the output the scan read an empty /proc,
    # where the assertion below holds however little is isolated.
    assert "cat /proc/" in stdout
    assert marker not in stdout
    assert (other_exit_code, other_timed_out) == (0, False)


async def _sandbox_runs_here(cwd: str) -> bool:
    """Whether this host can run the sandbox at all.

    Docker's default seccomp profile and Ubuntu's AppArmor userns restriction
    stop bwrap from creating namespaces, and the sandbox fails closed there.  A
    host that grants namespaces but masks ``/proc`` runs no command at all, so
    that one is a failure rather than a skip.
    """
    if not has_full_sandbox():
        return False
    _, stderr, exit_code, _ = await run_sandboxed(["true"], cwd)
    if PROC_MOUNT_REFUSED in stderr:
        pytest.fail(stderr)
    return exit_code == 0


async def _until_exists(path: Path, sandbox: "asyncio.Task") -> None:
    """Wait for *path*, failing if *sandbox* ends before it appears."""
    for _ in range(200):
        if path.exists():
            return
        if sandbox.done():
            pytest.fail(f"the other chat ended before it started: {sandbox.result()}")
        await asyncio.sleep(0.05)
    pytest.fail(f"the other chat never created {path.name}")
