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
