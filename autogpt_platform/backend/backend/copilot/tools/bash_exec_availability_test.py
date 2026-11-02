from unittest.mock import patch

import pytest

from backend.copilot.sdk.env import config as chat_config

from .bash_exec import BashExecTool


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
