import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.experts.models import Expert, ExpertWorkflowRef
from backend.api.features.experts.workflow_state import WorkflowValidationEvidence

from . import expert_tool_disabled_groups, get_available_tools
from .install_expert_workflow import InstallExpertWorkflowTool
from .models import (
    ErrorResponse,
    ExecutionStartedResponse,
    ExpertWorkflowInstalledResponse,
)

_MODULE = "backend.copilot.tools.install_expert_workflow"


def test_flag_off_hides_workflow_management_tool():
    disabled = expert_tool_disabled_groups(experts_enabled=False, expert_id=None)
    names = {
        tool["function"]["name"]
        for tool in get_available_tools(disabled_groups=disabled)
    }

    assert "install_expert_workflow" not in names


def _expert(expert_id: str = "expert-1", workflows=None) -> Expert:
    return Expert(
        id=expert_id,
        name="Remy",
        avatar_url=None,
        role="Sales lead",
        tagline=None,
        bio=None,
        skills=[],
        identity="Sales lead",
        voice_preferences="Concise",
        boundaries="Ask before external actions",
        protected_soul_rules=[],
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=workflows or [],
    )


def _workflow(library_agent_id: str | None = "library-1") -> ExpertWorkflowRef:
    return ExpertWorkflowRef(
        id="workflow-1",
        store_listing_version_id=None if library_agent_id else "listing-1",
        library_agent_id=library_agent_id,
        graph_id="graph-1",
        name="Lead Research",
        description="Find qualified leads",
    )


def _session(expert_id: str | None = None, *, tested: bool = False) -> MagicMock:
    session = MagicMock()
    session.session_id = "session-1"
    session.expert_id = expert_id
    session.messages = []
    if tested:
        session.messages = [
            SimpleNamespace(
                role="assistant",
                tool_call_id=None,
                content=None,
                tool_calls=[
                    {
                        "id": "call-1",
                        "function": {
                            "name": "run_agent",
                            "arguments": json.dumps(
                                {"library_agent_id": "library-1", "dry_run": True}
                            ),
                        },
                    }
                ],
            ),
            SimpleNamespace(
                role="tool",
                tool_call_id="call-1",
                tool_calls=None,
                content=json.dumps(
                    {
                        "type": "agent_output",
                        "library_agent_id": "library-1",
                        "execution": {"status": "completed", "nodes_failed": []},
                    }
                ),
            ),
        ]

    def copy_session(*, deep: bool, update: dict):
        assert deep is True
        clone = MagicMock()
        clone.session_id = session.session_id
        clone.expert_id = update["expert_id"]
        return clone

    session.model_copy.side_effect = copy_session
    return session


def _db(expert: Expert | None = None, workflow: ExpertWorkflowRef | None = None):
    db = MagicMock()
    db.get_expert = AsyncMock(return_value=expert or _expert())
    db.get_passed_workflow_validation = AsyncMock()
    db.install_workflow = AsyncMock(return_value=workflow or _workflow(None))
    db.install_library_workflow = AsyncMock(return_value=workflow or _workflow())
    db.claim_workflow_schedule = AsyncMock(return_value=True)
    return db


def _wire(monkeypatch, db, *, enabled: bool = True, validated: bool = True) -> None:
    monkeypatch.setattr(f"{_MODULE}.experts_db", lambda: db)
    monkeypatch.setattr(
        f"{_MODULE}.is_feature_enabled", AsyncMock(return_value=enabled)
    )
    db.get_passed_workflow_validation.return_value = (
        WorkflowValidationEvidence(
            id="validation-1",
            graph_version=3,
            test_execution_id="dry-run-1",
            artifacts=[],
        )
        if validated
        else None
    )


async def _execute(session, **overrides):
    values = {
        "purpose": "Research leads repeatedly",
        "expected_inputs": "ICP and geography",
        "expected_outputs": "Verified leads",
        "cadence": "Every weekday",
    }
    values.update(overrides)
    return await InstallExpertWorkflowTool()._execute("user-1", session, **values)


@pytest.mark.asyncio
async def test_flag_off_rejects_execution(monkeypatch):
    db = _db()
    _wire(monkeypatch, db, enabled=False)

    response = await _execute(
        _session(), expert_id="expert-1", store_listing_version_id="listing-1"
    )

    assert isinstance(response, ErrorResponse)
    db.install_workflow.assert_not_awaited()


@pytest.mark.asyncio
async def test_autopilot_installs_marketplace_workflow_on_owned_expert(monkeypatch):
    db = _db()
    _wire(monkeypatch, db)

    response = await _execute(
        _session(), expert_id="expert-1", store_listing_version_id="listing-1"
    )

    assert isinstance(response, ExpertWorkflowInstalledResponse)
    db.install_workflow.assert_awaited_once()
    assert response.expert.id == "expert-1"


@pytest.mark.asyncio
async def test_expert_defaults_to_self_for_private_workflow(monkeypatch):
    db = _db()
    _wire(monkeypatch, db)

    response = await _execute(
        _session("expert-1", tested=True), library_agent_id="library-1"
    )

    assert isinstance(response, ExpertWorkflowInstalledResponse)
    db.install_library_workflow.assert_awaited_once()
    assert db.install_library_workflow.await_args.args[:3] == (
        "user-1",
        "expert-1",
        "library-1",
    )
    assert (
        db.install_library_workflow.await_args.kwargs["validation_graph_version"] == 3
    )
    assert (
        db.install_library_workflow.await_args.kwargs["validation_execution_id"]
        == "dry-run-1"
    )


@pytest.mark.asyncio
async def test_expert_cannot_install_on_teammate(monkeypatch):
    db = _db()
    _wire(monkeypatch, db)

    response = await _execute(
        _session("expert-1", tested=True),
        expert_id="expert-2",
        library_agent_id="library-1",
    )

    assert isinstance(response, ErrorResponse)
    assert "only their own" in response.message
    db.install_library_workflow.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_workflow_requires_completed_safe_test(monkeypatch):
    db = _db()
    _wire(monkeypatch, db, validated=False)

    response = await _execute(
        _session(tested=True),
        expert_id="expert-1",
        library_agent_id="library-1",
    )

    assert isinstance(response, ErrorResponse)
    assert "no successful validation" in response.message
    db.install_library_workflow.assert_not_awaited()


@pytest.mark.asyncio
async def test_artifact_contract_requires_workspace_delivery(monkeypatch):
    db = _db()
    _wire(monkeypatch, db)

    response = await _execute(
        _session("expert-1"),
        library_agent_id="library-1",
        delivery_target="message",
        artifact_outputs=["report"],
    )

    assert isinstance(response, ErrorResponse)
    assert "require workspace_files" in response.message
    db.install_library_workflow.assert_not_awaited()


@pytest.mark.asyncio
async def test_schedule_run_is_attributed_to_owning_expert(monkeypatch):
    workflow = _workflow()
    expert = _expert(workflows=[workflow])
    db = _db(expert, workflow)
    _wire(monkeypatch, db)
    run = AsyncMock(
        return_value=ExecutionStartedResponse(
            message="scheduled",
            session_id="session-1",
            execution_id="schedule-1",
            graph_id="graph-1",
            graph_name="Lead Research",
            library_agent_id="library-1",
            status="SCHEDULED",
        )
    )
    monkeypatch.setattr(f"{_MODULE}.RunAgentTool._execute", run)

    response = await _execute(
        _session(tested=True),
        expert_id="expert-1",
        library_agent_id="library-1",
        schedule_cron="0 9 * * 1-5",
        schedule_name="Weekday lead research",
        schedule_approved=True,
    )

    assert isinstance(response, ExpertWorkflowInstalledResponse)
    attributed_session = run.await_args.args[1]
    assert attributed_session.expert_id == "expert-1"
    db.claim_workflow_schedule.assert_awaited_once_with(
        "user-1", "expert-1", "workflow-1", "schedule-1", "0 9 * * 1-5"
    )
