"""Snapshot reuse preserves credential selection and grant filtering."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import SecretStr

from backend.copilot.credential_selection import CredentialPin, set_turn_credential_pins
from backend.copilot.tools.expert_scope import (
    CredentialScopeSnapshot,
    ungranted_credential_hint,
)
from backend.copilot.tools.utils import match_user_credentials_to_graph
from backend.data.model import (
    APIKeyCredentials,
    CredentialsFieldInfo,
    OAuth2Credentials,
)


def _key(credential_id: str) -> APIKeyCredentials:
    return APIKeyCredentials(
        id=credential_id, provider="github", api_key=SecretStr("t")
    )


def _oauth(credential_id: str, scopes: list[str]) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=credential_id, provider="github", access_token=SecretStr("t"), scopes=scopes
    )


def _graph(credential_type: str, scopes: list[str] | None = None) -> MagicMock:
    field = CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["github"],
            "credentials_types": [credential_type],
            "credentials_scopes": scopes,
            "is_auto_credential": False,
        },
        by_alias=True,
    )
    graph = MagicMock(id="test-graph")
    graph.regular_credentials_inputs = {
        "credentials": (field, {("node-1", "credentials")}, True)
    }
    return graph


async def test_pre_resolved_graph_credentials_honor_the_session_selection():
    """Snapshot reuse must preserve the account selected for this session."""
    saved = [_key("first-account"), _key("chosen-account")]
    with (
        patch("backend.copilot.tools.utils.IntegrationCredentialsManager") as manager,
        patch(
            "backend.copilot.tools.utils.selected_credentials",
            AsyncMock(return_value={"github": "chosen-account"}),
        ) as selections,
    ):
        matched, missing = await match_user_credentials_to_graph(
            "test-user",
            _graph("api_key"),
            None,
            "test-session",
            available_credentials=saved,
        )
    assert matched["credentials"].id == "chosen-account"
    assert missing == []
    manager.assert_not_called()
    selections.assert_awaited_once_with("test-session")


async def test_pre_resolved_graph_credentials_do_not_switch_a_scheduled_pin():
    """A cached snapshot cannot widen a schedule's pinned account selection."""
    saved = [_oauth("personal", ["repo"]), _oauth("work", ["read:user"])]

    async def turn():
        set_turn_credential_pins({"github": CredentialPin(id="work", title="Work")})
        with (
            patch(
                "backend.copilot.tools.utils.IntegrationCredentialsManager"
            ) as manager,
            patch(
                "backend.copilot.tools.utils.selected_credentials",
                AsyncMock(return_value={"github": "work"}),
            ),
        ):
            result = await match_user_credentials_to_graph(
                "test-user",
                _graph("oauth2", ["repo"]),
                session_id="test-session",
                available_credentials=saved,
            )
            manager.assert_not_called()
            return result

    matched, missing = await asyncio.create_task(turn())
    assert matched == {}
    assert len(missing) == 1


async def test_hint_reuses_snapshot_and_filters_missing_requirements():
    """A shared snapshot must not offer a grant the next run would reject."""
    scope = CredentialScopeSnapshot(
        owned=[
            _oauth("matching", ["repo"]),
            _oauth("narrow-scope", ["read:user"]),
            _key("wrong-type"),
            _oauth("granted-cred", ["repo"]),
        ],
        allowed_ids={"granted-cred"},
    )
    requirements = iter(
        [{"provider": "github", "types": ["oauth2"], "scopes": ["repo"]}]
    )
    with (
        patch(
            "backend.copilot.tools.expert_scope.IntegrationCredentialsManager"
        ) as manager,
        patch("backend.copilot.tools.expert_scope.experts_db") as experts,
    ):
        hint = await ungranted_credential_hint(
            "test-user", "expert-a", {"github"}, requirements, credential_scope=scope
        )
    assert "credential_id=matching" in hint
    assert "narrow-scope" not in hint
    assert "wrong-type" not in hint
    assert "granted-cred" not in hint
    manager.assert_not_called()
    experts.assert_not_called()
