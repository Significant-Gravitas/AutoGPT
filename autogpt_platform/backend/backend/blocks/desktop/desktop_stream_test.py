"""The live stream: its password never rests on the box, and a re-open hands
back the URL the user holds only while the box has kept running."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.desktop._api import (
    HOME_PATH,
    STREAM_PORT,
    VNC_PASSWORD_PATH,
    VNC_PORT,
    VNC_USER,
    DesktopSession,
)


def _session(
    listening: bool,
    vnc_listening: bool | None = None,
    x11vnc_running: bool | None = None,
) -> tuple[DesktopSession, AsyncMock]:
    """A box where noVNC (and, unless told otherwise, x11vnc with it) is (not)
    serving; ``run`` records every command.  *x11vnc_running* is root's own
    process, which by default is what holds the VNC port."""
    vnc_listening = listening if vnc_listening is None else vnc_listening
    up = {
        f":{STREAM_PORT} ": listening,
        f":{VNC_PORT} ": vnc_listening,
        f"pgrep -x -u {VNC_USER} x11vnc": (
            vnc_listening if x11vnc_running is None else x11vnc_running
        ),
    }
    run = AsyncMock()

    async def fake_run(command: str, **kwargs):
        if "grep -q" in command:
            # As the shell would: the check passes only if everything it
            # asks about is there.
            if all(is_up for asked, is_up in up.items() if asked in command):
                return MagicMock()
            raise RuntimeError("Command exited with code 1 and error:")
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
    assert not any(cmd.startswith(("pkill", "x11vnc")) for cmd, _ in _commands(run))


@pytest.mark.asyncio
async def test_reopen_restarts_a_stream_whose_x11vnc_died_behind_a_live_novnc():
    """noVNC's port alone says nothing: it keeps listening with no x11vnc
    behind it, and the same URL would open onto a broken stream."""
    session, run = _session(listening=True, vnc_listening=False)

    stream, password = await session.start_stream("issued-before")

    assert password != "issued-before" and len(password) == 16
    assert f"password={password}" in stream.url
    commands = [cmd for cmd, _ in _commands(run)]
    # The whole stack, so the surviving proxy does not keep the stream port.
    stopped = next(i for i, cmd in enumerate(commands) if cmd.startswith("pkill"))
    assert "ovnc_proxy" in commands[stopped] and "x11vnc" in commands[stopped]
    assert any(cmd.startswith("x11vnc") for cmd in commands[stopped + 1 :])
    assert any("novnc_proxy --vnc" in cmd for cmd in commands[stopped + 1 :])


@pytest.mark.asyncio
async def test_a_listener_on_the_vnc_port_does_not_pass_for_x11vnc():
    """With x11vnc gone the box's user can bind its port with a server of its
    own; only root's process, asked for as root, cannot be forged."""
    session, run = _session(listening=True, vnc_listening=True, x11vnc_running=False)

    _, password = await session.start_stream("issued-before")

    assert password != "issued-before"
    commands = _commands(run)
    assert any(cmd.startswith("x11vnc") for cmd, _ in commands)
    (check,) = [(cmd, user) for cmd, user in commands if "grep -q" in cmd]
    assert check[1] == VNC_USER


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
async def test_stopping_the_stream_kills_only_the_vnc_stack_and_as_root():
    session, run = _session(listening=True)

    await session.stop_stream()

    ((command, user),) = _commands(run)
    assert "pkill -x x11vnc" in command and "pkill -f '[n]ovnc_proxy'" in command
    # The display keeps running: a later open serves the same screen again.
    assert "Xvfb" not in command and "xfce" not in command
    assert user == VNC_USER


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


@pytest.mark.asyncio
async def test_an_x11vnc_that_dies_after_detaching_is_caught_before_novnc_starts():
    """``-bg`` reports success once it has forked; the port never opening is
    the only sign, and noVNC's own port would come up regardless."""
    run = AsyncMock()

    async def fake_run(command: str, **kwargs):
        if command.startswith("netstat"):
            raise RuntimeError("not listening")
        if command.startswith("tail"):
            return MagicMock(stdout="caught X11 error")
        return MagicMock()

    run.side_effect = fake_run
    sandbox = MagicMock()
    sandbox.commands.run = run
    with (
        patch("backend.blocks.desktop._api._READY_POLL_ATTEMPTS", 2),
        patch("backend.blocks.desktop._api._READY_POLL_SECONDS", 0),
        pytest.raises(RuntimeError, match="x11vnc did not start: caught X11 error"),
    ):
        await DesktopSession(sandbox).start_stream(None)
    commands = [cmd for cmd, _ in _commands(run)]
    assert not any("novnc_proxy --vnc" in cmd for cmd in commands)
    # Detached but never serving, it is still running: stopped, not left.
    started = next(i for i, cmd in enumerate(commands) if cmd.startswith("x11vnc"))
    assert any(cmd.startswith("pkill") for cmd in commands[started + 1 :])


@pytest.mark.asyncio
async def test_a_novnc_that_never_serves_takes_x11vnc_down_with_it():
    """Otherwise x11vnc keeps serving under a password nobody was handed."""
    run = AsyncMock()

    async def fake_run(command: str, **kwargs):
        if command.startswith("netstat") and "5900" not in command:
            raise RuntimeError("not listening")
        if command.startswith("tail"):
            return MagicMock(stdout="websockify: address in use")
        return MagicMock()

    run.side_effect = fake_run
    sandbox = MagicMock()
    sandbox.commands.run = run
    with (
        patch("backend.blocks.desktop._api._READY_POLL_ATTEMPTS", 2),
        patch("backend.blocks.desktop._api._READY_POLL_SECONDS", 0),
        pytest.raises(RuntimeError, match="noVNC did not start: websockify"),
    ):
        await DesktopSession(sandbox).start_stream(None)
    commands = [cmd for cmd, _ in _commands(run)]
    started = next(i for i, cmd in enumerate(commands) if cmd.startswith("x11vnc"))
    assert any(cmd.startswith("pkill") for cmd in commands[started + 1 :])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing, error",
    [("x11vnc", "x11vnc did not start"), ("cd /opt/noVNC", "noVNC did not start")],
    ids=["x11vnc", "novnc"],
)
async def test_a_failed_start_does_not_leave_the_password_on_the_box(failing, error):
    """x11vnc deletes the file only once it has read it; one that fails before
    that would leave the credential resting in root's directory."""
    run = AsyncMock()

    async def fake_run(command: str, **kwargs):
        if command.startswith(failing):
            raise RuntimeError("Command exited with code 1 and error:")
        return MagicMock(stdout="why")

    run.side_effect = fake_run
    sandbox = MagicMock()
    sandbox.commands.run = run
    with pytest.raises(RuntimeError, match=error):
        await DesktopSession(sandbox).start_stream(None)
    commands = _commands(run)
    failed = next(i for i, (cmd, _) in enumerate(commands) if cmd.startswith(failing))
    assert any(
        f"rm -f {VNC_PASSWORD_PATH}" in cmd and user == VNC_USER
        for cmd, user in commands[failed + 1 :]
    )
