"""What the swap proxy is told: a value only for a host it is bound to."""

import contextlib
import importlib.util
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.integration_creds import ProviderTokenUnavailable
from backend.copilot.providers import SUPPORTED_PROVIDERS
from backend.copilot.swap_credentials import (
    SwapCredential,
    _host_is_bound,
    get_swap_bindings,
    resolve_swap_credential,
)
from backend.util import e2b_network

_M = "backend.copilot.swap_credentials"
_GITHUB_HOSTS = ["github.com", "api.github.com", "uploads.github.com"]


@contextlib.contextmanager
def _token(value, by_credential=None):
    by_credential = by_credential or {}
    with (
        patch(f"{_M}.get_provider_token", AsyncMock(return_value=value)) as lookup,
        patch(
            f"{_M}.get_provider_tokens_by_credential",
            AsyncMock(return_value=dict(by_credential)),
        ),
    ):
        yield lookup


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
    lookup.assert_awaited_once_with("user-1", "github", strict=True)


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


@pytest.mark.asyncio
async def test_a_lookup_that_fails_is_an_error_not_a_user_without_a_token():
    """The proxy scrubs responses against this answer, and refuses what it
    cannot scrub only when the backend says it cannot answer."""
    failing = AsyncMock(side_effect=ProviderTokenUnavailable("github"))
    with patch(f"{_M}.get_provider_token", failing):
        with pytest.raises(ProviderTokenUnavailable):
            await resolve_swap_credential("user-1", "github", "github.com")


@pytest.mark.asyncio
async def test_each_stored_credential_is_an_entry_under_its_id():
    """``hsurr:github:<id>`` is how a box names the account the chat picked."""
    with _token("ghp_a", {"cred-a": "ghp_a", "cred-b": "ghp_b"}):
        credential = await resolve_swap_credential("user-1", "github", "github.com")
    assert credential is not None
    assert credential.values == {
        "access_token": "ghp_a",
        "cred-a": "ghp_a",
        "cred-b": "ghp_b",
    }


@pytest.mark.asyncio
async def test_an_unbound_host_does_not_list_the_users_credentials():
    with _token("ghp_a", {"cred-a": "ghp_a"}):
        assert await resolve_swap_credential("user-1", "github", "evil.test") is None


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


# Mirrored in swap_proxy/swap_proxy/swap_test.py; keep the two identical.  The
# binding is checked on both sides of the wire, by two functions that cannot
# import each other and must not disagree.
HOST_BINDING_TABLE = [
    # (host, bound entries, may the credential go there?)
    ("api.github.com", ["api.github.com"], True),
    ("API.GitHub.com", ["api.github.com"], True),
    ("api.github.com", ["API.GITHUB.COM"], True),
    ("api.github.com:443", ["api.github.com"], True),
    ("github.com", ["api.github.com"], False),
    ("api.github.com.evil.test", ["api.github.com"], False),
    ("evilapi.github.com", ["api.github.com"], False),
    ("notgithub.com", ["github.com"], False),
    ("sub.github.com", ["github.com"], False),  # exact names do not cover subdomains
    ("raw.githubusercontent.com", [".githubusercontent.com"], True),
    ("a.b.githubusercontent.com", [".githubusercontent.com"], True),
    ("githubusercontent.com", [".githubusercontent.com"], False),
    ("evilgithubusercontent.com", [".githubusercontent.com"], False),
    ("githubusercontent.com.evil.test", [".githubusercontent.com"], False),
    ("api.github.com", [], False),
    ("", ["api.github.com"], False),
    ("api.github.com", ["github.com", "api.github.com"], True),
]


@pytest.mark.parametrize("host, entries, bound", HOST_BINDING_TABLE)
def test_host_binding_is_the_table_the_proxy_is_held_to(host, entries, bound):
    assert _host_is_bound(host, entries) is bound


def test_the_proxys_own_copy_agrees_on_every_host():
    """``swap.py`` imports only the standard library, so it loads by path even
    though its package does not install here.  Outside a full checkout there
    is nothing to load, and the mirrored table in its own suite stands alone."""
    path = Path(__file__).parents[3] / "swap_proxy" / "swap_proxy" / "swap.py"
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location("swap_proxy_swap", path)
    assert spec is not None and spec.loader is not None
    swap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(swap)
    for host, entries, _ in HOST_BINDING_TABLE:
        assert swap.host_in_list(host, entries) is _host_is_bound(host, entries), host
