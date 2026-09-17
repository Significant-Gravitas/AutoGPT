"""The live stream: its password never rests on the box, and a re-open hands
back the URL the user holds only while the box has kept running."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks.desktop._api import (
    HOME_PATH,
    VNC_PASSWORD_PATH,
    VNC_USER,
    DesktopSession,
)


def _session(listening: bool) -> tuple[DesktopSession, AsyncMock]:
    """A box where noVNC is (not) serving; ``run`` records every command."""
    run = AsyncMock()

    async def fake_run(command: str, **kwargs):
        if command.startswith("netstat") and "grep -q" in command:
            if listening:
                return MagicMock()
            raise RuntimeError("nothing on the stream port")
        return MagicMock()

    run.side_effect = fake_run
    sandbox = MagicMock()
    sandbox.sandbox_id = "sb-1"
    sandbox.commands.run = run
    sandbox.get_host = MagicMock(return_value="6080-sb-1.e2b.app")
    return DesktopSession(sandbox), run


def _commands(run: AsyncMock) -> list[tuple[str, str | None]]:
    return [(c.args[0], c.kwargs.get("user")) for c in run.await_args_list]


@pytest.mark.asyncio
async def test_a_fresh_stream_keeps_its_password_out_of_the_shells_reach():
    session, run = _session(listening=False)

    stream, password = await session.start_stream(None)

    assert len(password) == 16 and f"password={password}" in stream.url
    commands = _commands(run)
    vnc = [(cmd, user) for cmd, user in commands if "x11vnc" in cmd or "novnc" in cmd]
    # The whole VNC stack runs as root, not as the model's user.
    assert vnc and all(user == VNC_USER for _, user in vnc)
    # The password is handed to x11vnc through a root-only file it deletes
    # once read; nothing under the model's home ever holds it.
    (write,) = [cmd for cmd, _ in commands if password in cmd]
    assert "umask 077" in write and VNC_PASSWORD_PATH in write
    assert HOME_PATH not in write
    (x11vnc,) = [cmd for cmd, _ in commands if cmd.startswith("x11vnc")]
    assert f"-passwdfile rm:{VNC_PASSWORD_PATH}" in x11vnc
    assert "-storepasswd" not in x11vnc and password not in x11vnc
    # Root's x11vnc and the user's Xvfb cannot share memory: with MIT-SHM on,
    # x11vnc exits 1 (BadAccess on ShmAttach) and no desktop ever opens.
    assert " -noshm " in x11vnc
    # Root's logs stay out of /tmp, where the box's user could own a file of
    # the same name (a box that once ran its stream as the user does) and the
    # kernel would refuse root the open.
    (novnc,) = [cmd for cmd, _ in commands if "novnc_proxy --vnc" in cmd]
    assert "/tmp/" not in x11vnc and "/tmp/" not in novnc


@pytest.mark.asyncio
async def test_reopen_reuses_the_issued_password_while_novnc_still_serves():
    session, run = _session(listening=True)

    stream, password = await session.start_stream("issued-before")

    assert password == "issued-before" and "password=issued-before" in stream.url
    # Restarting the stack would sever the stream the user is watching.
    assert not any("x11vnc" in cmd for cmd, _ in _commands(run))


@pytest.mark.asyncio
async def test_without_a_remembered_password_the_stack_is_restarted():
    """noVNC still serving under a password nobody remembers (the box was
    paused since) is restarted: whoever held the old URL is locked out."""
    session, run = _session(listening=True)

    _, password = await session.start_stream(None)

    assert len(password) == 16
    assert any(cmd.startswith("pkill") for cmd, _ in _commands(run))
    assert any(cmd.startswith("x11vnc") for cmd, _ in _commands(run))


@pytest.mark.asyncio
async def test_a_remembered_password_is_dropped_once_the_proxy_is_gone():
    session, run = _session(listening=False)

    _, password = await session.start_stream("issued-before")

    assert password != "issued-before"
    assert any(cmd.startswith("x11vnc") for cmd, _ in _commands(run))


@pytest.mark.asyncio
async def test_a_failed_x11vnc_says_why_in_its_own_words():
    """Its stderr goes to a file in the box, so the bare exception is empty."""
    run = AsyncMock()

    async def fake_run(command: str, **kwargs):
        if command.startswith("netstat"):
            raise RuntimeError("nothing on the stream port")
        if command.startswith("x11vnc"):
            raise RuntimeError("Command exited with code 1 and error:")
        if command.startswith("tail"):
            return MagicMock(stdout="X Error of failed request:  BadAccess\n")
        return MagicMock()

    run.side_effect = fake_run
    sandbox = MagicMock()
    sandbox.commands.run = run
    with pytest.raises(RuntimeError, match="x11vnc did not start: X Error.*BadAccess"):
        await DesktopSession(sandbox).start_stream(None)
    (tail,) = [(c, u) for c, u in _commands(run) if c.startswith("tail")]
    assert tail[1] == VNC_USER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tail, said",
    [
        (MagicMock(stdout="  \n"), "(empty log)"),
        (RuntimeError("gone"), "(log unreadable)"),
    ],
    ids=["empty", "unreadable"],
)
async def test_a_failed_x11vnc_with_no_log_to_show_still_raises(tail, said):
    run = AsyncMock()

    async def fake_run(command: str, **kwargs):
        if command.startswith(("netstat", "x11vnc")):
            raise RuntimeError("Command exited with code 1 and error:")
        if command.startswith("tail"):
            if isinstance(tail, Exception):
                raise tail
            return tail
        return MagicMock()

    run.side_effect = fake_run
    sandbox = MagicMock()
    sandbox.commands.run = run
    with pytest.raises(RuntimeError, match="x11vnc did not start") as raised:
        await DesktopSession(sandbox).start_stream(None)
    assert said in str(raised.value)
