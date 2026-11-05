"""Fixtures shared by every copilot test."""

import os

import pytest

from backend.copilot import chat_env
from backend.copilot.config import ChatConfig


@pytest.fixture(autouse=True)
def _no_bans(mocker):
    """No owner is banned unless a test says so. Without this the lookup goes
    to a DatabaseManager these tests don't run, and waits out its retries."""
    for module in (
        "backend.copilot.turn_queue",
        "backend.copilot.executor.processor",
        "backend.copilot.briefing.generate",
    ):
        mocker.patch(
            f"{module}.is_user_banned", new=mocker.AsyncMock(return_value=False)
        )


@pytest.fixture(autouse=True)
def _no_cli_prelaunch(mocker):
    """No turn spawns a real CLI ahead of its query unless a test says so: a
    mocked ``ClaudeSDKClient`` would take the process and never stop it.
    Tests of the prelaunch patch ``service.CLIPrelaunch`` back themselves."""
    mocker.patch("backend.copilot.sdk.service.CLIPrelaunch", _NoCLIPrelaunch)


class _NoCLIPrelaunch:
    @staticmethod
    async def start(options, log_prefix):
        return None


@pytest.fixture
def clean_chat_settings(monkeypatch):
    """A ChatConfig built in the test sees only the settings the test sets:
    no chat setting or integration key from the machine (CI sets
    OPENAI_API_KEY, which dev's settings read as an OpenRouter key), and no
    dotenv file."""
    for name in list(os.environ):
        if name.startswith("CHAT_") or name in chat_env.MIGRATION_SETTING_NAMES:
            monkeypatch.delenv(name)
    monkeypatch.setitem(ChatConfig.model_config, "env_file", None)
    monkeypatch.setattr(chat_env, "backend_dotenv_paths", lambda: [])
    yield
    for name in list(os.environ):
        if name.startswith("CHAT_") or name in chat_env.MIGRATION_SETTING_NAMES:
            os.environ.pop(name, None)
