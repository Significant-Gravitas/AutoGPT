"""The platform's own keys are one kind, wherever they were registered.

The built-in list in ``credentials_store`` predates the block SDK, which registers
its providers' platform keys as ``"{provider}-default"``. Code asking "is this
ours?" must get the same answer for both (SECRT-3207).
"""

from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials
from backend.integrations.credentials_store import (
    SYSTEM_CREDENTIAL_IDS,
    is_system_credential,
    is_system_provider,
    system_providers,
)


def _sdk_default(provider: str) -> APIKeyCredentials:
    return APIKeyCredentials(
        id=f"{provider}-default",
        provider=provider,
        api_key=SecretStr("key"),
        title=f"{provider} API Key",
    )


def test_a_built_in_system_credential_is_ours():
    assert is_system_credential(sorted(SYSTEM_CREDENTIAL_IDS)[0])


def test_the_sdk_key_of_a_runnable_provider_is_ours():
    assert is_system_credential("firecrawl-default")
    assert is_system_credential("dataforseo-default")


def test_the_sdk_key_of_a_refused_provider_is_not_offered_as_ours():
    """Its key is refused wherever it is used, so nothing may count on it."""
    assert not is_system_credential("linear-default")


def test_a_users_own_credential_is_not_ours():
    assert not is_system_credential("3f2c9a1e-user-credential")


def test_providers_with_a_runnable_sdk_key_count_as_supplied():
    registered = [_sdk_default("firecrawl"), _sdk_default("linear")]
    with patch(
        "backend.sdk.registry.AutoRegistry.get_all_credentials",
        return_value=registered,
    ):
        providers = system_providers()
        assert is_system_provider("firecrawl")
        assert not is_system_provider("linear")
    assert "firecrawl" in providers
    assert "linear" not in providers


class _Secrets:
    """Settings secrets with only the given keys set."""

    def __init__(self, **keys: str) -> None:
        self._keys = keys

    def __getattr__(self, name: str) -> str:
        return self._keys.get(name, "")


def _system_providers_with(**keys: str) -> set[str]:
    with (
        patch(
            "backend.integrations.credentials_store.settings",
            SimpleNamespace(secrets=_Secrets(**keys)),
        ),
        patch(
            "backend.sdk.registry.AutoRegistry.get_all_credentials",
            return_value=[],
        ),
    ):
        return system_providers()


def test_a_built_in_provider_without_a_platform_key_is_not_supplied():
    """A keyless install supplies only the always-available Ollama credential,
    the same set ``get_all_creds`` hands out (#16772)."""
    assert _system_providers_with() == {"ollama"}


def test_a_built_in_provider_with_a_platform_key_is_supplied():
    assert _system_providers_with(openai_api_key="sk-test") == {"ollama", "openai"}
