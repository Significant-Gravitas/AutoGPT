"""The platform's own keys are one kind, wherever they were registered.

The built-in list in ``credentials_store`` predates the block SDK, which registers
its providers' platform keys as ``"{provider}-default"``. Code asking "is this
ours?" must get the same answer for both (SECRT-3207).
"""

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
