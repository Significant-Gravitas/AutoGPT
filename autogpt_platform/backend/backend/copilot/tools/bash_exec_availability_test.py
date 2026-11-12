from unittest.mock import patch

import pytest

from backend.copilot.context import turn_sandbox
from backend.copilot.sdk.env import config as chat_config

from .bash_exec import BashExecTool
from .local_pc_shim import LocalPCShim


@pytest.mark.parametrize(
    "platform,bwrap,e2b_enabled,e2b_key,available",
    [
        ("Windows", True, True, None, False),
        ("Darwin", True, True, None, False),
        ("Linux", False, True, None, False),
        ("Linux", True, False, None, True),
        ("Windows", False, True, "test-e2b-key", True),
        ("Darwin", False, True, "test-e2b-key", True),
        ("Windows", False, False, "test-e2b-key", False),
        ("Darwin", False, False, "test-e2b-key", False),
        ("Linux", False, False, "test-e2b-key", False),
        ("Linux", False, True, "test-e2b-key", True),
        ("Linux", True, True, None, True),
        ("Linux", True, False, "test-e2b-key", True),
        ("Linux", True, True, "test-e2b-key", True),
        ("Windows", False, True, "", False),
        ("Linux", True, True, "", True),
    ],
)
def test_only_offered_with_a_supported_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    bwrap: bool,
    e2b_enabled: bool,
    e2b_key: str | None,
    available: bool,
):
    monkeypatch.setattr(chat_config, "use_e2b_sandbox", e2b_enabled)
    monkeypatch.setattr(chat_config, "e2b_api_key", e2b_key)
    with (
        patch("backend.copilot.tools.sandbox._BWRAP_AVAILABLE", None),
        patch("backend.copilot.tools.sandbox.platform.system", return_value=platform),
        patch(
            "backend.copilot.tools.sandbox.shutil.which",
            return_value="/usr/bin/bwrap" if bwrap else None,
        ),
    ):
        assert BashExecTool().is_available is available


def test_a_local_pc_chat_offers_bash_without_a_server_sandbox(
    monkeypatch: pytest.MonkeyPatch,
):
    """A Local PC runs commands on the user's computer. With neither E2B nor
    bubblewrap on the server, bash was hidden from Local PC chats too, though
    the companion can run them; the chat hides it when the companion offers
    no shell."""
    monkeypatch.setattr(chat_config, "use_e2b_sandbox", False)
    monkeypatch.setattr(chat_config, "e2b_api_key", None)
    shim = LocalPCShim.__new__(LocalPCShim)
    with (
        patch("backend.copilot.tools.sandbox._BWRAP_AVAILABLE", None),
        patch("backend.copilot.tools.sandbox.platform.system", return_value="Darwin"),
    ):
        assert BashExecTool().is_available is False
        with turn_sandbox(shim):
            assert BashExecTool().is_available is True
