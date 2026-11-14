"""Each bash_exec sandbox keeps its own processes and IPC objects, and runs
under its resource limits."""

import asyncio
import re
from pathlib import Path

import pytest

from backend.copilot.tools.sandbox import (
    _RESOURCE_LIMITS,
    _build_bwrap_command,
    explain_proc_mount_failure,
    has_full_sandbox,
    run_sandboxed,
)

# bwrap's wording when it is refused a namespace, the one case worth skipping.
_NAMESPACE_REFUSED = "No permissions to create a new namespace"


def test_bwrap_gives_each_sandbox_its_own_processes_and_ipc():
    command = _build_bwrap_command(["true"], "/tmp/copilot-test", {"PATH": "/bin"})

    bwrap = command.index("bwrap")
    flags = command[bwrap : command.index("--", bwrap)]

    assert "--unshare-pid" in flags
    assert "--unshare-ipc" in flags
    assert "--unshare-net" in flags


def test_limits_and_oom_score_are_set_without_a_shell():
    command = _build_bwrap_command(["echo", "hi"], "/tmp/copilot-test", {})

    bwrap = command.index("bwrap")
    flags = command[bwrap : command.index("--", bwrap)]
    sandboxed = command[command.index("--", bwrap) + 1 :]
    assert command[:bwrap] == ["choom", "-n", "1000", "--"]
    assert flags.index("--remount-ro") > flags.index("--proc")
    assert flags[flags.index("--remount-ro") + 1] == "/proc"
    # No memory cap: the OOM order bounds memory, and a fixed cap breaks
    # encoders whose use grows with the host's cores.
    assert sandboxed == [
        "prlimit",
        "--nproc=256",
        "--fsize=4294967296",
        "--nofile=1024",
        "--",
        "echo",
        "hi",
    ]


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


# The tests below run real sandboxes, and skip only where bubblewrap cannot
# create namespaces at all.


@pytest.mark.asyncio
async def test_the_command_runs_under_its_limits_and_cannot_lower_its_oom_score(
    tmp_path,
):
    if not await _sandbox_runs_here(str(tmp_path)):
        pytest.skip("bubblewrap cannot create namespaces on this host")

    stdout, stderr, exit_code, _ = await run_sandboxed(
        [
            "sh",
            "-c",
            "cat /proc/self/limits; echo 999 > /proc/self/oom_score_adj;"
            " cat /proc/self/oom_score_adj",
        ],
        str(tmp_path),
    )

    assert exit_code == 0, stderr
    names = {
        "--nproc": "Max processes",
        "--fsize": "Max file size",
        "--nofile": "Max open files",
    }
    expected = {
        names[flag]: (value, value)
        for flag, value in (arg.split("=") for arg in _RESOURCE_LIMITS if "=" in arg)
    }
    in_force = {
        match[1]: (match[2], match[3])
        for match in re.finditer(r"^(Max [a-z ]+?)\s{2,}(\S+)\s+(\S+)", stdout, re.M)
    }
    assert {name: in_force.get(name) for name in expected} == expected
    assert stdout.splitlines()[-1] == "1000"


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
    stop bwrap from creating namespaces, and the sandbox fails closed there.
    Every other failure is this suite's to report rather than skip, including a
    host that grants namespaces but masks ``/proc``, where no command runs.
    """
    if not has_full_sandbox():
        return False
    _, stderr, exit_code, _ = await run_sandboxed(["true"], cwd)
    if exit_code == 0:
        return True
    if _NAMESPACE_REFUSED in stderr:
        return False
    pytest.fail(stderr or f"the sandbox exited {exit_code}")


async def _until_exists(path: Path, sandbox: "asyncio.Task") -> None:
    """Wait for *path*, failing if *sandbox* ends before it appears."""
    for _ in range(200):
        if path.exists():
            return
        if sandbox.done():
            pytest.fail(f"the other chat ended before it started: {sandbox.result()}")
        await asyncio.sleep(0.05)
    pytest.fail(f"the other chat never created {path.name}")
