"""Tests for the RFC 8628 device-auth base handler.

These cover the shared token-lifecycle helpers, which every device-code
provider inherits — the parts most likely to be relied on silently.
"""

import time
from typing import ClassVar

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.device_base import (
    BaseDeviceAuthHandler,
    DeviceAuthInitiation,
    DeviceAuthPollResult,
)


class _StubHandler(BaseDeviceAuthHandler):
    PROVIDER_NAME: ClassVar[str] = "stub_provider"
    DEFAULT_SCOPES: ClassVar[list[str]] = ["default:one", "default:two"]

    def __init__(self) -> None:
        self.refresh_calls = 0

    async def initiate_device_auth(self, scopes: list[str]) -> DeviceAuthInitiation:
        raise NotImplementedError

    async def poll_for_tokens(self, device_code: str) -> DeviceAuthPollResult:
        raise NotImplementedError

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        self.refresh_calls += 1
        credentials.access_token = SecretStr("refreshed-token")
        credentials.access_token_expires_at = int(time.time()) + 3600
        return credentials

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        return True


def make_credentials(
    expires_at: int | None, provider: str = "stub_provider"
) -> OAuth2Credentials:
    return OAuth2Credentials(
        provider=provider,
        access_token=SecretStr("current-token"),
        refresh_token=SecretStr("refresh-token"),
        access_token_expires_at=expires_at,
        scopes=["default:one"],
        title="Stub credentials",
    )


def test_needs_refresh_is_false_well_before_expiry():
    handler = _StubHandler()
    assert handler.needs_refresh(make_credentials(int(time.time()) + 3600)) is False


def test_needs_refresh_is_true_inside_the_skew_window():
    """Refresh early: a token expiring mid-request is as bad as an expired one."""
    handler = _StubHandler()
    assert handler.needs_refresh(make_credentials(int(time.time()) + 60)) is True


def test_needs_refresh_is_false_without_an_expiry():
    """A credential with no known expiry must not be refreshed on every call."""
    handler = _StubHandler()
    assert handler.needs_refresh(make_credentials(None)) is False


@pytest.mark.asyncio
async def test_refresh_tokens_rejects_a_foreign_provider():
    """A handler must never mint tokens for a provider it doesn't own."""
    handler = _StubHandler()
    foreign = make_credentials(int(time.time()) + 60, provider="someone_else")
    with pytest.raises(ValueError, match="cannot refresh tokens"):
        await handler.refresh_tokens(foreign)
    assert handler.refresh_calls == 0


@pytest.mark.asyncio
async def test_get_access_token_refreshes_only_when_needed():
    handler = _StubHandler()

    fresh = await handler.get_access_token(make_credentials(int(time.time()) + 3600))
    assert fresh == "current-token"
    assert handler.refresh_calls == 0

    stale = await handler.get_access_token(make_credentials(int(time.time()) + 60))
    assert stale == "refreshed-token"
    assert handler.refresh_calls == 1


def test_handle_default_scopes_fills_in_only_when_empty():
    handler = _StubHandler()
    assert handler.handle_default_scopes([]) == ["default:one", "default:two"]
    assert handler.handle_default_scopes(["explicit"]) == ["explicit"]
