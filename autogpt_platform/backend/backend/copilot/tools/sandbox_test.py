import asyncio

import pytest

from backend.copilot.tools.sandbox import (
    _build_bwrap_command,
    has_full_sandbox,
    run_sandboxed,
)


def _isolation_flags(command: list[str]) -> list[str]:
    return command[: command.index("--")]


def test_bwrap_gives_each_sandbox_its_own_processes_and_ipc():
    command = _build_bwrap_command(["true"], "/tmp/copilot-test", {"PATH": "/bin"})

    flags = _isolation_flags(command)

    assert "--unshare-pid" in flags
    assert "--unshare-ipc" in flags
    assert "--unshare-net" in flags


async def _bwrap_can_create_namespaces(cwd: str) -> bool:
    # Docker's default seccomp profile and Ubuntu's AppArmor userns restriction
    # both stop bwrap from creating namespaces; the sandbox then fails closed.
    if not has_full_sandbox():
        return False
    _, _, exit_code, _ = await run_sandboxed(["true"], cwd)
    return exit_code == 0


@pytest.mark.asyncio
async def test_sandbox_cannot_read_another_sandboxes_command_line(tmp_path):
    other = tmp_path / "other"
    scanner = tmp_path / "scanner"
    other.mkdir()
    scanner.mkdir()
    if not await _bwrap_can_create_namespaces(str(scanner)):
        pytest.skip("bubblewrap cannot create namespaces on this host")

    marker = "Bearer ghp_another_chats_token"
    other_chat = asyncio.create_task(
        run_sandboxed(["sh", "-c", "sleep 5", "other-chat", marker], str(other))
    )
    await asyncio.sleep(1)
    try:
        stdout, _, exit_code, _ = await run_sandboxed(
            [
                "sh",
                "-c",
                'cat /proc/[0-9]*/cmdline 2>/dev/null | tr "\\0" " "',
            ],
            str(scanner),
        )
    finally:
        await other_chat

    assert exit_code == 0
    assert marker not in stdout
