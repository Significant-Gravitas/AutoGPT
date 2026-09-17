"""Tests for chat tools utility functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.http import SendAuthenticatedWebRequestBlock
from backend.data.model import (
    APIKeyCredentials,
    CredentialsFieldInfo,
    HostScopedCredentials,
    OAuth2Credentials,
)
from backend.integrations.credentials_store import openai_credentials


def _make_regular_field() -> CredentialsFieldInfo:
    return CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["github"],
            "credentials_types": ["api_key"],
            "is_auto_credential": False,
        },
        by_alias=True,
    )


def test_build_missing_credentials_excludes_auto_creds():
    """
    build_missing_credentials_from_graph() should use regular_credentials_inputs
    and thus exclude auto_credentials from the "missing" set.
    """
    from backend.copilot.tools.utils import build_missing_credentials_from_graph

    regular_field = _make_regular_field()

    mock_graph = MagicMock()
    # regular_credentials_inputs should only return the non-auto field
    mock_graph.regular_credentials_inputs = {
        "github_api_key": (regular_field, {("node-1", "credentials")}, True),
    }

    result = build_missing_credentials_from_graph(mock_graph, matched_credentials=None)

    # Should include the regular credential
    assert "github_api_key" in result
    # Should NOT include the auto_credential (not in regular_credentials_inputs)
    assert "google_oauth2" not in result


@pytest.mark.asyncio
async def test_match_user_credentials_excludes_auto_creds():
    """
    match_user_credentials_to_graph() should use regular_credentials_inputs
    and thus exclude auto_credentials from matching.
    """
    from backend.copilot.tools.utils import match_user_credentials_to_graph

    regular_field = _make_regular_field()

    mock_graph = MagicMock()
    mock_graph.id = "test-graph"
    # regular_credentials_inputs returns only non-auto fields
    mock_graph.regular_credentials_inputs = {
        "github_api_key": (regular_field, {("node-1", "credentials")}, True),
    }

    # Mock the credentials manager to return no credentials
    with patch(
        "backend.copilot.tools.utils.IntegrationCredentialsManager"
    ) as MockCredsMgr:
        mock_store = AsyncMock()
        mock_store.get_all_creds.return_value = []
        MockCredsMgr.return_value.store = mock_store

        matched, missing = await match_user_credentials_to_graph(
            user_id="test-user", graph=mock_graph
        )

    # No credentials available, so github should be missing
    assert len(matched) == 0
    assert len(missing) == 1
    assert "github_api_key" in missing[0]


_SECRET = "sk-live-SHOULD-NOT-APPEAR"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "GET https://api.example.com/me?api_key=sk-live-abc failed",
            "GET https://api.example.com/me failed",
        ),
        ("Authorization: Bearer sk-live-abc rejected", "[redacted] rejected"),
        ("bad token=sk-live-abc", "bad [redacted]"),
        ("password: hunter2", "[redacted]"),
        ("HTTP 401 Error:\n  Unauthorized", "HTTP 401 Error: Unauthorized"),
        # The rejection reason is the point of the card, so prose that merely
        # mentions a secret-ish word must survive intact.
        ("Invalid API key provided.", "Invalid API key provided."),
        ("The token has expired; reconnect.", "The token has expired; reconnect."),
        ("Your basic plan does not allow this", "Your basic plan does not allow this"),
    ],
)
def test_sanitize_provider_message_drops_secrets(raw: str, expected: str):
    from backend.copilot.tools.utils import sanitize_provider_message

    assert sanitize_provider_message(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        f"api_key={_SECRET}",
        f'{{"api_key": "{_SECRET}"}}',
        f'{{"api_key":"{_SECRET}"}}',
        f'{{"access_token": "{_SECRET}"}}',
        f"{{'refresh_token': '{_SECRET}'}}",
        f'headers={{"X-Api-Key": "{_SECRET}"}}',
        f"Authorization: Bearer {_SECRET}",
        f"Authorization: Basic {_SECRET}",
        f"Authorization: Token {_SECRET}",
        f'{{"authorization": "Bearer {_SECRET}"}}',
    ],
)
def test_sanitize_provider_message_leaves_no_secret(raw: str):
    """Asserts on the secret, not on "[redacted]" — the Authorization shapes
    substituted the scheme and left the token standing next to the marker."""
    from backend.copilot.tools.utils import sanitize_provider_message

    assert _SECRET not in sanitize_provider_message(raw)


def test_sanitize_provider_message_is_bounded():
    from backend.copilot.tools.utils import sanitize_provider_message

    out = sanitize_provider_message("x" * 500, max_chars=50)
    assert out == "x" * 50 + "…"


def test_credential_rejection_status_reads_through_the_cause_chain():
    """Blocks wrap the provider's error, so only ``__cause__`` still has it."""
    from backend.copilot.tools.utils import credential_rejection_status
    from backend.util.exceptions import BlockUnknownError
    from backend.util.request import HTTPClientError

    try:
        try:
            raise HTTPClientError("Unauthorized", 401)
        except HTTPClientError as inner:
            raise BlockUnknownError("failed", "Block", "block-id") from inner
    except BlockUnknownError as wrapped:
        assert credential_rejection_status(wrapped) == 401


@pytest.mark.parametrize(
    "exc, expected",
    [
        (type("Aiohttp", (Exception,), {"status": 401})(), 401),
        # A 403 is a scope decision or a WAF, not a rejected credential.
        (type("Forbidden", (Exception,), {"status_code": 403})(), None),
        (
            type(
                "Requests",
                (Exception,),
                {"response": type("R", (), {"status_code": 401})()},
            )(),
            401,
        ),
        (type("ServerErr", (Exception,), {"status_code": 500})(), None),
        (ValueError("no status anywhere"), None),
    ],
)
def test_credential_rejection_status_only_matches_rejections(exc, expected):
    from backend.copilot.tools.utils import credential_rejection_status

    assert credential_rejection_status(exc) == expected


async def test_a_block_run_in_an_expert_session_uses_only_a_granted_credential():
    matched, missing = await _resolve_for("expert-a", ["granted-cred"])
    assert matched["credentials"].id == "granted-cred"
    assert missing == []


async def test_an_ungranted_credential_surfaces_as_missing_not_as_a_match():
    matched, missing = await _resolve_for("expert-a", [])
    assert matched == {}
    assert len(missing) == 1


async def test_a_plain_session_keeps_every_account_credential():
    matched, _ = await _resolve_for(None, [])
    assert matched["credentials"].id == "ungranted-cred"


def _host_cred(cred_id: str) -> HostScopedCredentials:
    return HostScopedCredentials(
        id=cred_id,
        provider="http",
        host="api.example.com",
        headers={"Authorization": SecretStr("Bearer token")},
        title=cred_id,
    )


async def _resolve_for(expert_id: str | None, allowed: list[str]):
    """Resolve an authenticated-request block against two account credentials
    for the same host, only one of which the expert has been granted."""
    from backend.copilot.tools.helpers import resolve_block_credentials

    experts = MagicMock()
    experts.expert_allowed_credential_ids = AsyncMock(return_value=allowed)
    with (
        patch(
            "backend.copilot.tools.utils.IntegrationCredentialsManager"
        ) as MockCredsMgr,
        patch("backend.data.db_accessors.experts_db", return_value=experts),
    ):
        MockCredsMgr.return_value.store = AsyncMock()
        MockCredsMgr.return_value.store.get_all_creds.return_value = [
            # Oldest first, as the store lists them. The ungranted one is the
            # newest, so a plain session picking it proves nothing was filtered.
            _host_cred("granted-cred"),
            _host_cred("ungranted-cred"),
        ]
        return await resolve_block_credentials(
            "test-user",
            SendAuthenticatedWebRequestBlock(),
            {"url": "https://api.example.com/v1/data"},
            expert_id,
        )


def _api_key(cred_id: str, provider: str = "github") -> APIKeyCredentials:
    return APIKeyCredentials(
        id=cred_id, provider=provider, title=cred_id, api_key=SecretStr("k")
    )


def _oauth(cred_id: str, scopes: list[str]) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=cred_id,
        provider="github",
        title=cred_id,
        access_token=SecretStr("t"),
        refresh_token=None,
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=scopes,
    )


def find_matching_credential(creds, field):
    from backend.copilot.tools.utils import find_matching_credential as find

    return find(creds, field)


def test_newest_of_the_users_own_credentials_wins():
    # The store lists credentials oldest first.
    picked = find_matching_credential(
        [_api_key("older"), _api_key("newer")], _make_regular_field()
    )
    assert picked is not None and picked.id == "newer"


def test_newest_that_has_the_scopes_wins_over_a_newer_one_without_them():
    field = CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["github"],
            "credentials_types": ["oauth2"],
            "credentials_scopes": ["repo", "read:org"],
        },
        by_alias=True,
    )
    picked = find_matching_credential(
        [
            _oauth("oldest", ["repo", "read:org"]),
            _oauth("middle", ["repo", "read:org"]),
            _oauth("newest-but-short", ["repo"]),
        ],
        field,
    )
    assert picked is not None and picked.id == "middle"


def test_own_credential_beats_a_system_one_listed_after_it():
    field = CredentialsFieldInfo.model_validate(
        {"credentials_provider": ["openai"], "credentials_types": ["api_key"]},
        by_alias=True,
    )
    own = _api_key("own-openai", provider="openai")
    picked = find_matching_credential([own, openai_credentials], field)
    assert picked is own


def test_system_credential_is_the_fallback():
    field = CredentialsFieldInfo.model_validate(
        {"credentials_provider": ["openai"], "credentials_types": ["api_key"]},
        by_alias=True,
    )
    assert find_matching_credential([openai_credentials], field) is openai_credentials
    assert find_matching_credential([], field) is None
