from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.desktop._api import DesktopSession
from backend.util.sandbox_login import LoginChainChanged


def _session_with_fresh_box() -> tuple[DesktopSession, AsyncMock]:
    """A sandbox where neither the window manager nor the X server is running yet."""
    run = AsyncMock()

    async def fake_run(command: str, **kwargs):
        if command.startswith("pgrep -x xfwm4"):
            # Not running until XFCE has been started.
            if not any("startxfce4" in c.args[0] for c in run.await_args_list):
                raise RuntimeError("no xfwm4")
            return MagicMock()
        if command.startswith("xdpyinfo"):
            if not any("Xvfb" in c.args[0] for c in run.await_args_list):
                raise RuntimeError("no X server")
            return MagicMock()
        return MagicMock()

    run.side_effect = fake_run
    sandbox = MagicMock()
    sandbox.commands.run = run
    return DesktopSession(sandbox), run


@pytest.mark.asyncio
async def test_a_refused_check_surfaces_instead_of_timing_out():
    """A changed login chain is a refusal, not a desktop that is still starting."""
    refused = AsyncMock(side_effect=LoginChainChanged("/etc/profile changed"))
    with patch("backend.blocks.desktop._api.run_internal", refused):
        with pytest.raises(LoginChainChanged):
            await DesktopSession(MagicMock()).ensure_display(1280, 720)
    refused.assert_awaited_once()


class TestEnsureDisplay:
    @pytest.mark.asyncio
    async def test_display_daemons_never_write_to_the_dropped_command_pipes(self):
        """Xvfb / XFCE run as background commands whose SDK handle is discarded.

        Their stdout and stderr must go to files: once the handle is gone the
        pipe is closed and the next warning Xvfb logs kills it with SIGPIPE,
        taking every X client (the whole desktop) down with it.
        """
        session, run = _session_with_fresh_box()

        await session.ensure_display(1280, 720)

        background = [
            c.args[0] for c in run.await_args_list if c.kwargs.get("background")
        ]
        xvfb = next(c for c in background if c.startswith("Xvfb"))
        xfce = next(c for c in background if "startxfce4" in c)
        assert "1280x720x24" in xvfb
        # XFCE reads its settings from the real home, which the login skip hides.
        assert xfce.startswith("export HOME=/home/user;")
        for command in (xvfb, xfce):
            assert "> /tmp/" in command and "2>&1" in command, command

    @pytest.mark.asyncio
    async def test_running_window_manager_short_circuits(self):
        session, run = _session_with_fresh_box()
        run.side_effect = None  # every check succeeds: xfwm4 is already up

        await session.ensure_display(1280, 720)

        assert run.await_count == 1
        assert run.await_args.args[0].startswith("pgrep -x xfwm4")
