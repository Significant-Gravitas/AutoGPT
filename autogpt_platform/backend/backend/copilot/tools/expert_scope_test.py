from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.model import ChatSession
from backend.copilot.tools.expert_scope import (
    ExpertWorkflowScope,
    install_saved_agent,
    require_installed_workflow,
    resolve_target_expert,
    ungranted_credential_hint,
)
from backend.copilot.tools.models import AgentSavedResponse, ErrorResponse

_PATH = "backend.copilot.tools.expert_scope"


def _expert(expert_id: str, graph_ids: list[str] = ()):
    return MagicMock(
        id=expert_id,
        workflows=[
            MagicMock(library_agent_id=f"lib-{g}", graph_id=g) for g in graph_ids
        ],
    )


@pytest.fixture
def experts():
    db = MagicMock()
    db.get_expert = AsyncMock(
        side_effect=lambda user_id, expert_id, **_: (
            _expert(expert_id, ["installed"]) if expert_id == "expert-a" else None
        )
    )
    db.install_workflow = AsyncMock()
    db.expert_allowed_credential_ids = AsyncMock(return_value=["granted-cred"])
    db.settle_credential_seed = AsyncMock()
    with patch(f"{_PATH}.experts_db", return_value=db):
        yield db


def _expert_session(expert_id: str | None = "expert-a") -> ChatSession:
    return ChatSession.new("user-1", dry_run=False, expert_id=expert_id)


def test_scope_allows_only_installed_workflows():
    scope = ExpertWorkflowScope(
        expert_id="e", library_agent_ids=["lib-1"], graph_ids=["graph-1"]
    )
    assert scope.allows_graph("graph-1")
    assert not scope.allows_graph("graph-2")
    assert scope.allows_agent(library_agent_id="lib-1", graph_id=None)
    assert scope.allows_agent(library_agent_id="other", graph_id="graph-1")
    assert not scope.allows_agent(library_agent_id="other", graph_id="graph-2")


async def test_expert_is_refused_uninstalled_workflow(experts):
    error = await require_installed_workflow(
        "user-1", _expert_session(), graph_id="other", name="Other"
    )
    assert isinstance(error, ErrorResponse)
    assert error.error == "workflow_not_installed"
    assert "install_expert_workflow" in error.message


async def test_expert_may_use_installed_workflow(experts):
    assert (
        await require_installed_workflow(
            "user-1", _expert_session(), graph_id="installed", name="x"
        )
        is None
    )
    assert (
        await require_installed_workflow(
            "user-1",
            _expert_session(),
            graph_id="lib-installed",
            library_agent_id="lib-installed",
            name="x",
        )
        is None
    )


async def test_personal_autopilot_may_use_any_workflow(experts):
    assert (
        await require_installed_workflow(
            "user-1", _expert_session(None), graph_id="anything", name="x"
        )
        is None
    )
    experts.get_expert.assert_not_awaited()


async def test_missing_expert_fails_closed(experts):
    error = await require_installed_workflow(
        "user-1", _expert_session("expert-gone"), graph_id="installed", name="x"
    )
    assert isinstance(error, ErrorResponse)


async def test_expert_session_targets_itself_only(experts):
    assert await resolve_target_expert("user-1", _expert_session(), None) == "expert-a"
    assert (
        await resolve_target_expert("user-1", _expert_session(), "expert-a")
        == "expert-a"
    )
    denied = await resolve_target_expert("user-1", _expert_session(), "expert-b")
    assert isinstance(denied, ErrorResponse) and denied.error == "access_denied"


async def test_personal_autopilot_must_name_a_real_expert(experts):
    missing = await resolve_target_expert("user-1", _expert_session(None), None)
    assert isinstance(missing, ErrorResponse) and missing.error == "expert_required"
    unknown = await resolve_target_expert("user-1", _expert_session(None), "nope")
    assert isinstance(unknown, ErrorResponse) and unknown.error == "expert_not_found"
    assert (
        await resolve_target_expert("user-1", _expert_session(None), "expert-a")
        == "expert-a"
    )


def _saved() -> AgentSavedResponse:
    return AgentSavedResponse(
        message="Saved.",
        agent_id="graph-new",
        agent_name="New",
        library_agent_id="lib-new",
        library_agent_link="/library/agents/lib-new",
        agent_page_link="/build?flowID=graph-new",
    )


async def test_agent_built_by_expert_is_installed_on_it(experts):
    result = await install_saved_agent("user-1", _expert_session(), _saved())
    experts.install_workflow.assert_awaited_once_with(
        "user-1", "expert-a", library_agent_id="lib-new"
    )
    assert isinstance(result, AgentSavedResponse)
    assert "Installed on this expert" in result.message


async def test_failed_install_is_reported_not_raised(experts):
    experts.install_workflow.side_effect = RuntimeError("boom")
    result = await install_saved_agent("user-1", _expert_session(), _saved())
    assert "install_expert_workflow" in result.message


async def test_personal_autopilot_build_is_not_installed_anywhere(experts):
    result = await install_saved_agent("user-1", _expert_session(None), _saved())
    experts.install_workflow.assert_not_awaited()
    assert result.message == "Saved."


def test_provider_slug_uses_the_wire_value_of_enums():
    from backend.copilot.tools.expert_scope import provider_slug
    from backend.integrations.providers import ProviderName

    assert provider_slug(ProviderName.GITHUB) == "github"
    assert provider_slug("github") == "github"


async def test_hint_lists_owned_but_ungranted_credentials(experts):
    store = MagicMock()
    store.get_all_creds = AsyncMock(
        return_value=[
            MagicMock(id="granted-cred", provider="github", title="GH granted"),
            MagicMock(id="spare-cred", provider="github", title="GH spare"),
            MagicMock(id="other-cred", provider="slack", title="Slack"),
        ]
    )
    with patch(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        return_value=MagicMock(store=store),
    ):
        hint = await ungranted_credential_hint("user-1", "expert-a", {"github"})
        none = await ungranted_credential_hint("user-1", "expert-a", {"notion"})
        personal = await ungranted_credential_hint("user-1", None, {"github"})
    assert "spare-cred" in hint and "GH spare" in hint
    assert "granted-cred" not in hint and "other-cred" not in hint
    assert "grant_expert_credential" in hint
    assert none == "" and personal == ""


async def test_missing_credentials_are_annotated_with_expert_grants(experts):
    from backend.copilot.tools.expert_scope import annotate_expert_grants

    def cred(id: str, type: str, scopes: list[str]):
        return MagicMock(id=id, provider="github", title=id, type=type, scopes=scopes)

    store = MagicMock()
    store.get_all_creds = AsyncMock(
        return_value=[
            cred("granted-cred", "oauth2", ["repo"]),
            cred("spare-cred", "oauth2", ["repo"]),
            cred("wrong-type", "api_key", []),
            cred("narrow-scope", "oauth2", ["read:user"]),
        ]
    )
    missing = {
        "github_credentials": {
            "provider": "github",
            "types": ["oauth2"],
            "scopes": ["repo"],
        }
    }
    with patch(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        return_value=MagicMock(store=store),
    ):
        annotated = await annotate_expert_grants("user-1", "expert-a", missing)
        untouched = await annotate_expert_grants("user-1", None, missing)
    grant = annotated["github_credentials"]["expert_grant"]
    assert grant["expert_id"] == "expert-a"
    assert [c["id"] for c in grant["credentials"]] == ["spare-cred"]
    assert grant["credentials"][0]["type"] == "oauth2"
    assert "expert_grant" not in missing["github_credentials"]
    assert untouched is missing


async def test_grant_candidates_must_match_the_requested_mcp_server(experts):
    from backend.copilot.tools.expert_scope import annotate_expert_grants
    from backend.data.model import OAuth2Credentials

    def mcp_cred(id: str, server_url: str):
        return OAuth2Credentials(
            id=id,
            provider="mcp",
            title=id,
            access_token=SecretStr("token"),
            scopes=[],
            metadata={"mcp_server_url": server_url},
        )

    store = MagicMock()
    store.get_all_creds = AsyncMock(
        return_value=[
            mcp_cred("right-server", "https://mcp.example.com/sse"),
            mcp_cred("other-server", "https://mcp.other.com/sse"),
        ]
    )
    missing = {
        "mcp_credentials": {
            "provider": "mcp",
            "types": ["oauth2"],
            "discriminator_values": ["https://mcp.example.com/sse"],
        }
    }
    with patch(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        return_value=MagicMock(store=store),
    ):
        annotated = await annotate_expert_grants("user-1", "expert-a", missing)
    grant = annotated["mcp_credentials"]["expert_grant"]
    assert [c["id"] for c in grant["credentials"]] == ["right-server"]


async def test_agent_built_by_expert_settles_grants_before_install(experts):
    order: list[str] = []
    experts.settle_credential_seed.side_effect = lambda *_: order.append("settle")
    experts.install_workflow.side_effect = lambda *_, **__: order.append("install")
    await install_saved_agent("user-1", _expert_session(), _saved())
    assert order == ["settle", "install"]
