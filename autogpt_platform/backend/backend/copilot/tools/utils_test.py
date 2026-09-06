"""Tests for chat tools utility functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.model import CredentialsFieldInfo


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
    ],
)
def test_sanitize_provider_message_drops_secrets(raw: str, expected: str):
    from backend.copilot.tools.utils import sanitize_provider_message

    assert sanitize_provider_message(raw) == expected


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
        (type("Aiohttp", (Exception,), {"status": 403})(), 403),
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
def test_credential_rejection_status_only_matches_auth_statuses(exc, expected):
    from backend.copilot.tools.utils import credential_rejection_status

    assert credential_rejection_status(exc) == expected
