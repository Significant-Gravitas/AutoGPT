"""Config-load guard for the Claude subscription route."""

import pytest

from backend.util.settings import BehaveAs

from .chat_env import ChatEnvironmentError, check_chat_environment
from .config import ChatConfig, settings
from .config_test import _ENV_VARS_TO_CLEAR


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


class TestClaudeSubscriptionRefusedOnCloud:
    """A Claude OAuth subscription is one person's credential serving every
    user of the deployment: it may only run self-hosted."""

    def test_refused_when_behaving_as_cloud(self, mocker):
        mocker.patch.object(settings.config, "behave_as", BehaveAs.CLOUD)
        assert ChatConfig(platform_route="subscription").platform_route == "disabled"
        with pytest.raises(ChatEnvironmentError, match="BEHAVE_AS=cloud"):
            check_chat_environment(
                {"CHAT_PLATFORM_ROUTE": "subscription", "BEHAVE_AS": "cloud"}, []
            )

    def test_refused_from_the_environment(self, mocker, monkeypatch):
        mocker.patch.object(settings.config, "behave_as", BehaveAs.CLOUD)
        monkeypatch.setenv("CHAT_PLATFORM_ROUTE", "subscription")
        assert ChatConfig().platform_route == "disabled"

    def test_allowed_on_a_self_hosted_deployment(self, mocker):
        mocker.patch.object(settings.config, "behave_as", BehaveAs.LOCAL)
        assert (
            ChatConfig(platform_route="subscription").platform_route == "subscription"
        )

    def test_other_routes_load_on_cloud(self, mocker):
        mocker.patch.object(settings.config, "behave_as", BehaveAs.CLOUD)
        cfg = ChatConfig(platform_route="openrouter", openrouter_api_key="or-key")
        assert cfg.platform_route == "openrouter"
