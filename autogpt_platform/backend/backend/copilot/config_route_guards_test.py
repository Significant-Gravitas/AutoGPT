"""Config-load guards for the Claude subscription route and the billing-header switch."""

import pytest
from pydantic import ValidationError

from backend.util.settings import BehaveAs

from .config import ChatConfig, settings
from .config_test import _ENV_VARS_TO_CLEAR, _make_direct_safe_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


class TestClaudeSubscriptionRefusedOnCloud:
    """A Claude OAuth subscription is one person's credential serving every
    user of the deployment: it may only run self-hosted."""

    def test_refused_when_behaving_as_cloud(self, mocker):
        mocker.patch.object(settings.config, "behave_as", BehaveAs.CLOUD)
        with pytest.raises(ValidationError, match="BEHAVE_AS=cloud"):
            _make_direct_safe_config(use_claude_code_subscription=True)

    def test_refused_from_the_environment(self, mocker, monkeypatch):
        mocker.patch.object(settings.config, "behave_as", BehaveAs.CLOUD)
        monkeypatch.setenv("CHAT_USE_CLAUDE_CODE_SUBSCRIPTION", "true")
        with pytest.raises(ValidationError, match="self-hosted deployment"):
            _make_direct_safe_config()

    def test_allowed_on_a_self_hosted_deployment(self, mocker):
        mocker.patch.object(settings.config, "behave_as", BehaveAs.LOCAL)
        cfg = _make_direct_safe_config(use_claude_code_subscription=True)
        assert cfg.effective_transport == "subscription"

    def test_other_routes_load_on_cloud(self, mocker):
        mocker.patch.object(settings.config, "behave_as", BehaveAs.CLOUD)
        cfg = _make_direct_safe_config(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.effective_transport == "openrouter"


class TestAttributionHeaderSwitch:
    """Off by default (the CLI env is unchanged) and read from the env."""

    def test_defaults_off(self):
        field = ChatConfig.model_fields["claude_agent_omit_attribution_header"]
        assert field.default is False

    def test_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("CHAT_CLAUDE_AGENT_OMIT_ATTRIBUTION_HEADER", "true")
        assert ChatConfig().claude_agent_omit_attribution_header is True
