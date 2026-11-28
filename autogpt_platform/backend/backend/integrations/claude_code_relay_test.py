"""A relay token stands for one user's one credential until it is revoked, is
stored only as a digest, and the relay is offered only at an outside address."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from backend.data import redis_client
from backend.data.redis_client import get_redis_async
from backend.integrations.claude_code_relay import (
    RELAY_UNREACHABLE,
    RelayGrant,
    RelayUnreachableError,
    mint_relay_token,
    relay_base_url,
    relay_grant,
    revoke_relay_token,
)
from backend.util.settings import Config
from backend.util.testing import is_tcp_port_reachable


@pytest.mark.parametrize(
    "platform_base_url, expected",
    [
        (
            "https://backend.example.com",
            "https://backend.example.com/api/anthropic-relay",
        ),
        ("https://example.com/_agpt/", "https://example.com/_agpt/api/anthropic-relay"),
        ("https://8.8.8.8:8006", "https://8.8.8.8:8006/api/anthropic-relay"),
    ],
)
def test_the_relay_is_offered_at_the_platforms_public_address(
    platform_base_url: str, expected: str
):
    assert _base_url_for(platform_base_url) == expected


@pytest.mark.parametrize(
    "platform_base_url",
    [
        "",
        "http://localhost:8006",
        "http://api.localhost",
        "http://127.0.0.1:8006",
        "http://192.168.1.20:8006",
        "http://[::1]:8006",
    ],
)
def test_an_address_outside_networks_cannot_reach_is_refused(platform_base_url: str):
    with pytest.raises(RelayUnreachableError, match="PLATFORM_BASE_URL"):
        _base_url_for(platform_base_url)
    assert "PLATFORM_BASE_URL" in RELAY_UNREACHABLE


def test_the_setting_the_refusal_names_turns_the_relay_off(
    monkeypatch: pytest.MonkeyPatch,
):
    assert Config.model_fields["claude_code_key_in_sandbox"].default is False
    monkeypatch.setenv("CLAUDE_CODE_KEY_IN_SANDBOX", "true")

    assert Config().claude_code_key_in_sandbox is True
    assert "CLAUDE_CODE_KEY_IN_SANDBOX=true" in RELAY_UNREACHABLE


@pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis to store tokens in",
)
async def test_a_token_stands_for_its_grant_until_it_is_revoked():
    token = await mint_relay_token("user-a", "cred-1", ttl_seconds=60)
    other = await mint_relay_token("user-b", "cred-2", ttl_seconds=60)
    try:
        assert await relay_grant(token) == RelayGrant(
            user_id="user-a", credentials_id="cred-1"
        )
        assert await relay_grant("not-a-token") is None

        await revoke_relay_token(token)

        assert await relay_grant(token) is None
        assert await relay_grant(other) == RelayGrant(
            user_id="user-b", credentials_id="cred-2"
        )
        redis = await get_redis_async()
        digest = hashlib.sha256(other.encode()).hexdigest()
        assert await redis.exists(f"anthropic-relay:token:{digest}")
        assert 0 < await redis.ttl(f"anthropic-relay:token:{digest}") <= 60
        assert not await redis.exists(f"anthropic-relay:token:{other}")
    finally:
        await revoke_relay_token(token)
        await revoke_relay_token(other)


def _base_url_for(platform_base_url: str) -> str:
    settings = MagicMock()
    settings.config.platform_base_url = platform_base_url
    with patch(
        "backend.integrations.claude_code_relay.Settings", return_value=settings
    ):
        return relay_base_url()
