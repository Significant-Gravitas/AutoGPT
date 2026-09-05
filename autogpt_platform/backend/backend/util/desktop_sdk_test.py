"""Failed desktop probes must respect a deadline and release the sandbox."""

from unittest.mock import MagicMock, patch

import pytest
from e2b import CommandExitException, TimeoutException

from backend.util.desktop_sdk import DesktopSandbox


@pytest.mark.parametrize("probe_error", [False, True])
def test_readiness_probe_times_out_without_spinning(probe_error):
    sandbox = MagicMock()
    sandbox.commands.run.return_value = MagicMock(exit_code=1)
    if probe_error:
        sandbox.commands.run.side_effect = CommandExitException(
            stderr="not listening", stdout="", exit_code=1, error=None
        )
    with (
        patch(
            "backend.util.desktop_sdk.time.monotonic",
            side_effect=[0, 0, 0, 0.5, 0.5, 1],
        ),
        patch("backend.util.desktop_sdk.time.sleep") as sleep,
        pytest.raises(TimeoutException, match="deadline"),
    ):
        DesktopSandbox._wait_and_verify(
            sandbox, "probe", lambda r: r.exit_code == 0, timeout=1
        )
    assert sandbox.commands.run.call_count == 2
    assert sleep.call_count == 2
    sandbox.kill.assert_called_once()


def test_successful_probe_keeps_sandbox_alive():
    sandbox = MagicMock()
    with patch("backend.util.desktop_sdk.time.monotonic", side_effect=[0, 0]):
        assert DesktopSandbox._wait_and_verify(sandbox, "probe", lambda r: True)
    sandbox.commands.run.assert_called_once_with("probe", timeout=10)
    sandbox.kill.assert_not_called()


def test_probe_cleanup_failure_preserves_timeout(caplog):
    sandbox = MagicMock()
    sandbox.kill.side_effect = RuntimeError("kill failed")
    with (
        patch("backend.util.desktop_sdk.time.monotonic", side_effect=[0, 1]),
        pytest.raises(TimeoutException),
    ):
        DesktopSandbox._wait_and_verify(sandbox, "probe", lambda r: False, timeout=1)
    assert caplog.records[-1].exc_info is not None
