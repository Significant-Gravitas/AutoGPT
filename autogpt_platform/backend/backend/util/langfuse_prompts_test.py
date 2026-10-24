from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from langfuse._client.resource_manager import LangfuseResourceManager

from backend.util import langfuse_prompts
from backend.util.langfuse_prompts import (
    get_langfuse_prompt_client,
    langfuse_prompts_configured,
)


@pytest.fixture(autouse=True)
def fresh_client() -> Iterator[None]:
    get_langfuse_prompt_client.cache_clear()
    LangfuseResourceManager.reset()
    try:
        yield
    finally:
        get_langfuse_prompt_client.cache_clear()
        LangfuseResourceManager.reset()


def _secrets(public_key: str = "pk-lf-test", secret_key: str = "sk-lf-test"):
    return MagicMock(
        langfuse_public_key=public_key,
        langfuse_secret_key=secret_key,
        langfuse_host="http://127.0.0.1:9",
    )


def test_the_client_is_built_from_settings_with_tracing_off():
    """Credentials read from ``.env`` reach ``Settings`` but not the process
    environment the SDK falls back to."""
    with (
        patch.object(langfuse_prompts, "Settings") as settings,
        patch.object(langfuse_prompts, "Langfuse") as langfuse,
    ):
        settings.return_value.secrets = _secrets()
        get_langfuse_prompt_client()

    langfuse.assert_called_once_with(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        host="http://127.0.0.1:9",
        tracing_enabled=False,
    )


def test_the_client_sets_up_no_tracer(monkeypatch: pytest.MonkeyPatch):
    """Even when the environment asks for tracing, no span processor is
    installed, so nothing is exported to Langfuse."""
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    with patch.object(langfuse_prompts, "Settings") as settings:
        settings.return_value.secrets = _secrets()
        client = get_langfuse_prompt_client()

    resources = client._resources
    assert resources is not None
    assert resources.tracing_enabled is False
    assert resources.tracer_provider is None
    assert client.api is not None


def test_the_client_is_built_once():
    with (
        patch.object(langfuse_prompts, "Settings") as settings,
        patch.object(langfuse_prompts, "Langfuse") as langfuse,
    ):
        settings.return_value.secrets = _secrets()
        assert get_langfuse_prompt_client() is get_langfuse_prompt_client()

    langfuse.assert_called_once()


@pytest.mark.parametrize(
    ("public_key", "secret_key", "configured"),
    [("pk", "sk", True), ("", "sk", False), ("pk", "", False)],
)
def test_prompts_are_configured_only_with_both_keys(
    public_key: str, secret_key: str, configured: bool
):
    with patch.object(langfuse_prompts, "Settings") as settings:
        settings.return_value.secrets = _secrets(public_key, secret_key)
        assert langfuse_prompts_configured() is configured
