"""What the swap proxy is told: a value only for a host it is bound to."""

import re
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.providers import SUPPORTED_PROVIDERS
from backend.copilot.swap_credentials import (
    SwapCredential,
    get_swap_bindings,
    resolve_swap_credential,
)
from backend.util import e2b_network

_M = "backend.copilot.swap_credentials"
_GITHUB_HOSTS = ["github.com", "api.github.com", "uploads.github.com"]


def _token(value):
    return patch(f"{_M}.get_provider_token", AsyncMock(return_value=value))


@pytest.mark.asyncio
async def test_bindings_carry_hosts_and_no_values():
    assert await get_swap_bindings() == {"github": _GITHUB_HOSTS}


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["api.github.com", "GitHub.com", "github.com:443"])
async def test_a_bound_host_gets_the_users_token(host):
    with _token("ghp_real") as lookup:
        credential = await resolve_swap_credential("user-1", "github", host)
    assert credential == SwapCredential(
        name="github",
        values={"access_token": "ghp_real"},
        allowed_hosts=_GITHUB_HOSTS,
    )
    lookup.assert_awaited_once_with("user-1", "github")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    [
        "evil.test",
        "api.github.com.evil.test",
        "notgithub.com",
        "raw.githubusercontent.com",  # serves GitHub content, takes no token
        "",
    ],
)
async def test_an_unbound_host_gets_nothing_and_the_token_is_never_fetched(host):
    with _token("ghp_real") as lookup:
        assert await resolve_swap_credential("user-1", "github", host) is None
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unknown_name_gets_nothing():
    with _token("ghp_real") as lookup:
        assert await resolve_swap_credential("user-1", "gitlab", "gitlab.com") is None
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_user_who_has_not_connected_the_provider_gets_nothing():
    with _token(None):
        assert await resolve_swap_credential("user-1", "github", "github.com") is None


def test_every_provider_says_where_its_token_may_go():
    """A new provider must decide this; an empty list means never swapped."""
    for slug, entry in SUPPORTED_PROVIDERS.items():
        assert isinstance(entry["swap_hosts"], list), slug
        for host in entry["swap_hosts"]:
            assert host == host.lower() and "/" not in host and "*" not in host, host


def test_minted_usernames_are_what_the_proxy_accepts():
    """The proxy (swap_proxy/owners.py) looks up nothing that does not match
    this shape; the two packages cannot import each other, so it is pinned here."""
    assert re.fullmatch(r"box-[0-9a-f]{16}", e2b_network._mint().username)
