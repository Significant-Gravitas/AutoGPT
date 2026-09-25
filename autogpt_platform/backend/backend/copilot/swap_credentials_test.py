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
    NoLiveBox,
    SwapCredential,
    _host_is_bound,
    get_swap_bindings,
    resolve_swap_credential,
)
from backend.util import e2b_network

_M = "backend.copilot.swap_credentials"
_GITHUB_HOSTS = [
    "github.com",
    "api.github.com",
    "uploads.github.com",
    "raw.githubusercontent.com",
    "gist.github.com",
]
_GITHUB_CONTENT = [".githubusercontent.com"]


_BOX = "box-0123456789abcdef"
_RECORD = {
    "owner": "session:s-1",
    "user_id": "user-1",
    "swaps": True,
    "sandbox_id": "sb-1",
}


@pytest.fixture(autouse=True)
def live_box():
    """The asking box's egress record: a live CoPilot box of user-1's."""
    record = dict(_RECORD)
    with patch(
        f"{_M}.credential_record", AsyncMock(side_effect=lambda box: record)
    ) as lookup:
        yield record
    for call in lookup.await_args_list:
        assert call.args == (_BOX,)


@contextlib.contextmanager
def _token(tokens, granted=("cred-a",)):
    """*tokens*: credential id to token (a string is every id's token), and
    the ids granted to sandbox sb-1."""

    async def token(user_id, name, credential_id=None, strict=False, lock=False):
        assert lock, "the swap service refreshes under the manager's lock"
        assert strict, "a failed lookup must reach the proxy as an outage"
        if isinstance(tokens, dict):
            return tokens.get(credential_id)
        return tokens

    async def grants(sandbox_id, name):
        return set(granted) if sandbox_id == "sb-1" else set()

    with (
        patch(f"{_M}.get_provider_token", AsyncMock(side_effect=token)) as lookup,
        patch(f"{_M}.granted_to_box", AsyncMock(side_effect=grants)),
    ):
        yield lookup


@pytest.mark.asyncio
async def test_bindings_carry_hosts_and_no_values():
    assert await get_swap_bindings() == {"github": _GITHUB_HOSTS + _GITHUB_CONTENT}


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["api.github.com", "GitHub.com", "github.com:443"])
async def test_a_bound_host_gets_the_credential_granted_to_the_box(host):
    with _token("ghp_real") as lookup:
        credential = await resolve_swap_credential("user-1", "github", host, _BOX)
    assert credential == SwapCredential(
        name="github",
        values={"cred-a": "ghp_real"},
        allowed_hosts=_GITHUB_HOSTS,
    )
    lookup.assert_awaited_once_with(
        "user-1", "github", credential_id="cred-a", strict=True, lock=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    [
        "evil.test",
        "api.github.com.evil.test",
        "notgithub.com",
        "githubusercontent.com.evil.test",
        "codeload.github.com",  # archives only: binary, nothing to scrub
        "",
    ],
)
async def test_an_unbound_host_gets_nothing_and_the_token_is_never_fetched(host):
    with _token("ghp_real") as lookup:
        assert await resolve_swap_credential("user-1", "github", host, _BOX) is None
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unknown_name_gets_nothing():
    with _token("ghp_real") as lookup:
        assert (
            await resolve_swap_credential("user-1", "gitlab", "gitlab.com", _BOX)
            is None
        )
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_granted_credential_that_yields_no_token_gets_nothing():
    with _token(None):
        assert (
            await resolve_swap_credential("user-1", "github", "github.com", _BOX)
            is None
        )


@pytest.mark.asyncio
async def test_a_lookup_that_fails_is_an_error_not_a_user_without_a_token():
    """The proxy scrubs responses against this answer, and refuses what it
    cannot scrub only when the backend says it cannot answer."""
    with _token("ghp_real"):
        with (
            patch(
                f"{_M}.get_provider_token",
                AsyncMock(side_effect=ProviderTokenUnavailable("github")),
            ),
            pytest.raises(ProviderTokenUnavailable),
        ):
            await resolve_swap_credential("user-1", "github", "github.com", _BOX)


@pytest.mark.asyncio
async def test_grants_that_cannot_be_read_are_an_error_too():
    with (
        patch(f"{_M}.granted_to_box", AsyncMock(side_effect=ConnectionError())),
        pytest.raises(ConnectionError),
    ):
        await resolve_swap_credential("user-1", "github", "github.com", _BOX)


@pytest.mark.asyncio
async def test_only_what_was_granted_to_the_box_resolves():
    """The user has accounts A and B; the chat picked B.  Typing A's id, or a
    bare ``hsurr:github``, gets nothing: A was never granted to this box."""
    with _token({"cred-a": "ghp_a", "cred-b": "ghp_b"}, granted=["cred-b"]) as lookup:
        credential = await resolve_swap_credential(
            "user-1", "github", "github.com", _BOX
        )
    assert credential is not None
    assert credential.values == {"cred-b": "ghp_b"}
    # Only the granted credential is looked up, not every stored one.
    lookup.assert_awaited_once_with(
        "user-1", "github", credential_id="cred-b", strict=True, lock=True
    )


@pytest.mark.asyncio
async def test_a_box_granted_nothing_gets_nothing(live_box):
    live_box["sandbox_id"] = "sb-other"
    with _token("ghp_real") as lookup:
        assert (
            await resolve_swap_credential("user-1", "github", "github.com", _BOX)
            is None
        )
    lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record",
    [
        None,  # revoked, rotated, or never minted
        {**_RECORD, "user_id": "user-2"},
        {**_RECORD, "swaps": False},  # a block's box
        {**_RECORD, "swaps": "true"},
    ],
    ids=["no record", "another user", "does not swap", "not an explicit true"],
)
async def test_only_a_live_box_of_the_users_that_swaps_gets_a_value(record):
    """The proxy names the box; the backend's own record decides.  A proxy
    that asks for a user with no live box of theirs gets nothing, and an
    error rather than "not connected", so that it refuses that connection's
    responses instead of passing them on unscrubbed."""
    with (
        patch(f"{_M}.credential_record", AsyncMock(return_value=record)),
        _token("ghp_real") as lookup,
        pytest.raises(NoLiveBox),
    ):
        await resolve_swap_credential("user-1", "github", "github.com", _BOX)
    lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    [
        "objects.githubusercontent.com",
        "gist.githubusercontent.com",
        "media.githubusercontent.com",
    ],
)
async def test_a_content_host_gets_the_values_to_scrub_but_may_not_be_sent_them(
    host,
):
    """What a user committed or pasted comes back from these; the proxy needs
    the values to scrub it, and must never swap one into a request there."""
    with _token("ghp_real"):
        credential = await resolve_swap_credential("user-1", "github", host, _BOX)
    assert credential is not None
    assert credential.values["cred-a"] == "ghp_real"
    assert credential.allowed_hosts == _GITHUB_HOSTS
    assert not _host_is_bound(host, credential.allowed_hosts)


_HOSTNAME = re.compile(r"\.?[a-z0-9-]+(\.[a-z0-9-]+)+")


def test_every_provider_says_where_its_token_may_go():
    """A new provider must decide this; an empty list means never swapped."""
    for slug, entry in SUPPORTED_PROVIDERS.items():
        assert isinstance(entry["swap_hosts"], list), slug
        assert isinstance(entry["content_hosts"], list), slug
        for host in [*entry["swap_hosts"], *entry["content_hosts"]]:
            # Names or a leading-dot suffix, as ``host_in_list`` reads them:
            # no scheme, port, path, wildcard or bare address.
            assert _HOSTNAME.fullmatch(host), (slug, host)
            assert not host.replace(".", "").isdigit(), (slug, host)
        # A content host takes no token: in both lists it would take one.
        overlap = [
            h
            for h in entry["content_hosts"]
            if _host_is_bound(h.lstrip("."), entry["swap_hosts"])
        ]
        assert overlap == [], slug


def test_no_host_is_bound_to_two_providers():
    """A host names one provider's credential; two would make the proxy fetch
    and scrub both, and leave which one a request meant to the box."""
    seen: dict[str, str] = {}
    for slug, entry in SUPPORTED_PROVIDERS.items():
        for host in [*entry["swap_hosts"], *entry["content_hosts"]]:
            assert host not in seen, (host, seen.get(host), slug)
            seen[host] = slug


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host",
    # A private repo's raw file with ``Authorization: token``; git over HTTPS
    # to a gist with Basic.
    ["raw.githubusercontent.com", "gist.github.com"],
)
async def test_the_github_content_hosts_that_take_the_token_may_be_sent_it(host):
    with _token("ghp_real"):
        credential = await resolve_swap_credential("user-1", "github", host, _BOX)
    assert credential is not None
    assert _host_is_bound(host, credential.allowed_hosts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "providers, sendable",
    [(None, True), (["github"], True), ([], False), (["linear"], False)],
    ids=["no ceiling", "github allowed", "none allowed", "github outside"],
)
async def test_a_provider_outside_the_boxs_ceiling_is_scrubbed_never_sent(
    live_box, providers, sendable
):
    live_box["providers"] = providers
    with _token("ghp_real"):
        credential = await resolve_swap_credential(
            "user-1", "github", "api.github.com", _BOX
        )
    assert credential is not None
    assert credential.values["cred-a"] == "ghp_real"
    assert credential.allowed_hosts == (_GITHUB_HOSTS if sendable else [])
