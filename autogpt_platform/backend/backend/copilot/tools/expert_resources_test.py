from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.experts.models import ExpertCredentialRef, ExpertWorkflowRef
from backend.copilot.model import ChatSession
from backend.copilot.tools.expert_resources import (
    ExpertCredentialsResponse,
    ExpertWorkflowResponse,
    GrantExpertCredentialTool,
    InstallExpertWorkflowTool,
    RemoveExpertWorkflowTool,
    RevokeExpertCredentialTool,
)
from backend.copilot.tools.models import ErrorResponse

_PATH = "backend.copilot.tools.expert_resources"
_SCOPE = "backend.copilot.tools.expert_scope"


def _ref():
    return ExpertWorkflowRef(
        id="wf-1",
        store_listing_version_id=None,
        library_agent_id="lib-1",
        graph_id="graph-1",
        name="Digest",
        description=None,
    )


def _cred():
    return ExpertCredentialRef(
        credential_id="cred-1", provider="github", title="GH", type="api_key"
    )


@pytest.fixture
def experts():
    db = MagicMock()
    db.get_expert = AsyncMock(
        side_effect=lambda user_id, expert_id, **_: (
            MagicMock(id=expert_id, workflows=[_ref()])
            if expert_id in {"expert-a", "expert-b"}
            else None
        )
    )
    db.install_workflow = AsyncMock(return_value=_ref())
    db.expert_allowed_credential_ids = AsyncMock(return_value=[])
    db.remove_workflow = AsyncMock()
    db.grant_expert_credentials = AsyncMock(return_value=[_cred()])
    db.revoke_expert_credential = AsyncMock(return_value=[])
    with (
        patch(f"{_PATH}.experts_db", return_value=db),
        patch(f"{_SCOPE}.experts_db", return_value=db),
    ):
        yield db


def _session(expert_id: str | None) -> ChatSession:
    return ChatSession.new("user-1", dry_run=False, expert_id=expert_id)


async def test_expert_installs_library_agent_onto_itself(experts):
    result = await InstallExpertWorkflowTool()._execute(
        "user-1", _session("expert-a"), library_agent_id="lib-1"
    )
    assert isinstance(result, ExpertWorkflowResponse)
    assert result.expert_id == "expert-a"
    experts.install_workflow.assert_awaited_once_with(
        "user-1", "expert-a", library_agent_id="lib-1", store_listing_version_id=None
    )


async def test_expert_cannot_install_onto_another_expert(experts):
    result = await InstallExpertWorkflowTool()._execute(
        "user-1", _session("expert-a"), library_agent_id="lib-1", expert_id="expert-b"
    )
    assert isinstance(result, ErrorResponse) and result.error == "access_denied"
    experts.install_workflow.assert_not_awaited()


async def test_autopilot_installs_marketplace_agent_by_slug(experts):
    details = MagicMock(store_listing_version_id="slv-9")
    with patch(
        f"{_PATH}.fetch_graph_from_store_slug",
        new=AsyncMock(return_value=(MagicMock(), details)),
    ):
        result = await InstallExpertWorkflowTool()._execute(
            "user-1",
            _session(None),
            username_agent_slug="creator/digest",
            expert_id="expert-b",
        )
    assert isinstance(result, ExpertWorkflowResponse)
    experts.install_workflow.assert_awaited_once_with(
        "user-1", "expert-b", library_agent_id=None, store_listing_version_id="slv-9"
    )


async def test_autopilot_must_name_the_expert(experts):
    result = await InstallExpertWorkflowTool()._execute(
        "user-1", _session(None), library_agent_id="lib-1"
    )
    assert isinstance(result, ErrorResponse) and result.error == "expert_required"


async def test_install_needs_exactly_one_source(experts):
    result = await InstallExpertWorkflowTool()._execute(
        "user-1",
        _session("expert-a"),
        library_agent_id="lib-1",
        username_agent_slug="a/b",
    )
    assert isinstance(result, ErrorResponse)
    experts.install_workflow.assert_not_awaited()


async def test_remove_by_library_agent_id(experts):
    result = await RemoveExpertWorkflowTool()._execute(
        "user-1", _session("expert-a"), library_agent_id="lib-1"
    )
    assert isinstance(result, ExpertWorkflowResponse)
    experts.remove_workflow.assert_awaited_once_with("user-1", "expert-a", "wf-1")


async def test_remove_unknown_workflow_is_refused(experts):
    result = await RemoveExpertWorkflowTool()._execute(
        "user-1", _session("expert-a"), workflow_id="nope"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "workflow_not_installed"
    experts.remove_workflow.assert_not_awaited()


async def test_autopilot_grants_and_revokes(experts):
    granted = await GrantExpertCredentialTool()._execute(
        "user-1", _session(None), expert_id="expert-a", credential_id="cred-1"
    )
    revoked = await RevokeExpertCredentialTool()._execute(
        "user-1", _session(None), expert_id="expert-a", credential_id="cred-1"
    )
    assert isinstance(granted, ExpertCredentialsResponse)
    assert isinstance(revoked, ExpertCredentialsResponse)
    experts.grant_expert_credentials.assert_awaited_once_with(
        "user-1", "expert-a", ["cred-1"]
    )
    experts.revoke_expert_credential.assert_awaited_once_with(
        "user-1", "expert-a", "cred-1"
    )


async def test_expert_cannot_grant_to_another_expert(experts):
    result = await GrantExpertCredentialTool()._execute(
        "user-1", _session("expert-a"), expert_id="expert-b", credential_id="cred-1"
    )
    assert isinstance(result, ErrorResponse) and result.error == "access_denied"
    experts.grant_expert_credentials.assert_not_awaited()


async def test_grant_rejects_foreign_credential(experts):
    experts.grant_expert_credentials.side_effect = ValueError("Not your credentials")
    result = await GrantExpertCredentialTool()._execute(
        "user-1", _session(None), expert_id="expert-a", credential_id="x"
    )
    assert isinstance(result, ErrorResponse) and "Not your" in result.message


async def test_list_expert_workflows_for_own_expert(experts):
    from backend.copilot.tools.expert_resources import (
        ExpertWorkflowsResponse,
        ListExpertWorkflowsTool,
    )

    result = await ListExpertWorkflowsTool()._execute("user-1", _session("expert-a"))
    assert isinstance(result, ExpertWorkflowsResponse)
    assert [w.id for w in result.workflows] == ["wf-1"]
    assert "wf-1" in result.message


async def test_autopilot_lists_an_experts_credentials(experts):
    from backend.copilot.tools.expert_resources import ListExpertCredentialsTool

    experts.list_expert_credentials = AsyncMock(return_value=[_cred()])
    result = await ListExpertCredentialsTool()._execute(
        "user-1", _session(None), expert_id="expert-b"
    )
    assert isinstance(result, ExpertCredentialsResponse)
    assert "cred-1" in result.message
    experts.list_expert_credentials.assert_awaited_once_with("user-1", "expert-b")


async def test_expert_requests_a_grant_and_leaves_a_pending_question(experts):
    from backend.copilot.tools.expert_resources import (
        CredentialGrantRequestedResponse,
        RequestCredentialGrantTool,
    )

    chat = MagicMock()
    chat.set_session_pending_question = AsyncMock()
    session = _session("expert-a")
    with patch(f"{_PATH}.chat_db", return_value=chat):
        result = await RequestCredentialGrantTool()._execute(
            "user-1", session, credential_id="cred-9", provider="github", reason="push"
        )
    assert isinstance(result, CredentialGrantRequestedResponse)
    assert session.metadata.pending_question is not None
    assert "cred-9" in session.metadata.pending_question.text
    chat.set_session_pending_question.assert_awaited_once()
    event = RequestCredentialGrantTool().activity_event(session, result)
    assert event is not None and event.category == "INTEGRATION"
    assert event.expert_id == "expert-a" and event.object_id == "cred-9"


async def test_personal_autopilot_cannot_request_a_grant(experts):
    from backend.copilot.tools.expert_resources import RequestCredentialGrantTool

    result = await RequestCredentialGrantTool()._execute(
        "user-1", _session(None), credential_id="cred-9"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "not_an_expert_session"


async def test_remove_rejects_conflicting_selectors(experts):
    result = await RemoveExpertWorkflowTool()._execute(
        "user-1", _session("expert-a"), workflow_id="wf-1", library_agent_id="lib-1"
    )
    assert isinstance(result, ErrorResponse)
    experts.remove_workflow.assert_not_awaited()


async def test_grant_request_fails_when_it_cannot_be_recorded(experts):
    from backend.copilot.tools.expert_resources import RequestCredentialGrantTool

    chat = MagicMock()
    chat.set_session_pending_question = AsyncMock(side_effect=RuntimeError("db"))
    session = _session("expert-a")
    with patch(f"{_PATH}.chat_db", return_value=chat):
        result = await RequestCredentialGrantTool()._execute(
            "user-1", session, credential_id="cred-9"
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "request_not_recorded"
    assert session.metadata.pending_question is None


async def test_expert_cannot_grant_itself(experts):
    result = await GrantExpertCredentialTool()._execute(
        "user-1", _session("expert-a"), expert_id="expert-a", credential_id="cred-1"
    )
    assert isinstance(result, ErrorResponse) and result.error == "access_denied"
    experts.grant_expert_credentials.assert_not_awaited()


async def test_expert_install_settles_its_grants_before_the_workflow_lands(experts):
    order: list[str] = []
    experts.expert_allowed_credential_ids.side_effect = (
        lambda *_: order.append("settle") or []
    )
    experts.install_workflow.side_effect = lambda *_, **__: (
        order.append("install") or _ref()
    )
    result = await InstallExpertWorkflowTool()._execute(
        "user-1", _session("expert-a"), library_agent_id="lib-1"
    )
    assert isinstance(result, ExpertWorkflowResponse)
    assert order == ["settle", "install"]
    experts.expert_allowed_credential_ids.assert_awaited_once_with("user-1", "expert-a")


async def test_autopilot_install_does_not_touch_the_experts_grants(experts):
    result = await InstallExpertWorkflowTool()._execute(
        "user-1", _session(None), expert_id="expert-a", library_agent_id="lib-1"
    )
    assert isinstance(result, ExpertWorkflowResponse)
    experts.expert_allowed_credential_ids.assert_not_awaited()
