"""Tests for chat tools utility functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.http import SendAuthenticatedWebRequestBlock
from backend.data.model import CredentialsFieldInfo, HostScopedCredentials


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
        # An OAuth token endpoint's error can quote these back.
        f"client_secret={_SECRET}",
        f'{{"client_secret": "{_SECRET}"}}',
        f'{{"id_token": "{_SECRET}"}}',
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
            _host_cred("ungranted-cred"),
            _host_cred("granted-cred"),
        ]
        return await resolve_block_credentials(
            "test-user",
            SendAuthenticatedWebRequestBlock(),
            {"url": "https://api.example.com/v1/data"},
            expert_id,
        )


def _key_cred(cred_id: str, provider: str = "github"):
    from backend.data.model import APIKeyCredentials

    return APIKeyCredentials(
        id=cred_id, provider=provider, title=cred_id, api_key=SecretStr("k")
    )


def _find(creds, selected=None, *, ask=False, field=None):
    from backend.copilot.tools.utils import find_matching_credential

    return find_matching_credential(
        creds, field or _make_regular_field(), selected, ask_when_ambiguous=ask
    )


def test_the_credential_the_user_picked_wins():
    creds = [_key_cred("work"), _key_cred("personal")]
    picked = _find(creds, {"github": "personal"}, ask=True)
    assert picked is not None and picked.id == "personal"


def test_a_chat_tool_asks_rather_than_choosing_between_two_accounts():
    # None surfaces as a missing credential, which is the card with a picker.
    assert _find([_key_cred("work"), _key_cred("personal")], ask=True) is None


def test_a_single_credential_needs_no_pick():
    picked = _find([_key_cred("only")], ask=True)
    assert picked is not None and picked.id == "only"


def test_a_pick_that_no_longer_fits_is_ignored_not_trusted():
    # Deleted, or for another provider: fall back to asking, never to a guess.
    creds = [_key_cred("work"), _key_cred("personal")]
    assert _find(creds, {"github": "deleted-id"}, ask=True) is None
    assert _find(creds, {"slack": "work"}, ask=True) is None


def test_callers_that_cannot_ask_keep_the_first_fit():
    # Schedules and expert setup have nobody to ask; their behaviour is unchanged.
    picked = _find([_key_cred("work"), _key_cred("personal")])
    assert picked is not None and picked.id == "work"


def test_a_system_credential_is_used_when_the_user_has_none_of_their_own():
    from backend.integrations.credentials_store import openai_credentials

    field = CredentialsFieldInfo.model_validate(
        {"credentials_provider": ["openai"], "credentials_types": ["api_key"]},
        by_alias=True,
    )
    assert _find([openai_credentials], ask=True, field=field) is openai_credentials
    own = _key_cred("own-openai", provider="openai")
    assert _find([own, openai_credentials], ask=True, field=field) is own
