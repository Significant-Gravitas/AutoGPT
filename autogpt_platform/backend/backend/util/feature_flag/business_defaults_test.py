import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.util.feature_flag as ff
from backend.api.features.graph_executions import activity_gate
from backend.api.features.home import service as home
from backend.api.features.onboarding_dump import routes as onboarding
from backend.api.features.store import skill_model, skill_routes
from backend.copilot.briefing.models import BriefingContent
from backend.copilot.graphiti import config as memory
from backend.data import user as user_db
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.model import GraphExecutionStats
from backend.executor import activity_status_generator as activity
from backend.util.exceptions import ExecutionFailureReason
from backend.util.models import Pagination
from backend.util.settings import FeatureFlagBackend


@pytest.fixture(autouse=True)
def unavailable_flag_provider(monkeypatch, mocker):
    for name in list(os.environ):
        if name.startswith(("FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_", "FORCE_ALL_")):
            monkeypatch.delenv(name)
    mocker.patch.object(
        ff.settings.config, "feature_flag_backend", FeatureFlagBackend.POSTHOG
    )
    mocker.patch.object(ff.posthog, "get_flag_client", return_value=None)
    mocker.patch.object(
        ff,
        "_fetch_user_context_status",
        side_effect=AssertionError("Unexpected lookup"),
    )


@pytest.fixture
def route_app():
    app = FastAPI()
    app.include_router(skill_routes.router, prefix="/store/skills")
    app.include_router(onboarding.router)
    return app


@pytest.mark.parametrize(
    ("gate", "flag"),
    [
        (memory.is_enabled_for_user, "GRAPHITI_MEMORY"),
        (memory.is_communities_enabled_for_user, "GRAPHITI_COMMUNITIES_ENABLED"),
    ],
)
async def test_memory_defaults_require_a_user_and_allow_operator_disable(
    gate, flag, monkeypatch
):
    assert await gate(None) is False
    assert await gate("") is False
    assert await gate("user-1") is True

    monkeypatch.setenv(f"FORCE_FLAG_{flag}", "false")
    assert await gate("user-1") is False


@pytest.mark.parametrize("disabled", [False, True])
def test_marketplace_browse_default_and_operator_disable(
    disabled, route_app, monkeypatch, mocker
):
    browse = mocker.patch.object(
        skill_routes.skill_db,
        "get_marketplace_skills",
        return_value=skill_model.MarketplaceSkillsResponse(
            skills=[], pagination=Pagination.empty()
        ),
    )
    if disabled:
        monkeypatch.setenv("FORCE_FLAG_SKILLS_HUB", "false")

    with TestClient(route_app) as client:
        response = client.get("/store/skills")

    assert response.status_code == (404 if disabled else 200)
    assert browse.await_count == (0 if disabled else 1)


@pytest.mark.parametrize("disabled", [False, True])
def test_onboarding_status_default_and_operator_disable(
    disabled, route_app, monkeypatch, mocker, test_user_id
):
    route_app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": test_user_id,
        "role": "user",
        "email": "test@example.com",
    }
    mocker.patch.object(user_db, "is_user_banned", return_value=False)
    get_dump = mocker.patch.object(onboarding.db, "get_dump", return_value=None)
    if disabled:
        monkeypatch.setenv("FORCE_FLAG_ONBOARDING_BRAIN_DUMP", "false")

    with TestClient(route_app) as client:
        response = client.get("/onboarding/brain-dump/status")

    assert response.status_code == (404 if disabled else 200)
    if disabled:
        get_dump.assert_not_awaited()
    else:
        get_dump.assert_awaited_once_with(test_user_id)


def test_default_on_routes_still_require_authentication(route_app, mocker):
    install = mocker.patch.object(skill_routes.skill_db, "install_marketplace_skill")
    get_dump = mocker.patch.object(onboarding.db, "get_dump")

    with TestClient(route_app) as client:
        assert client.post("/store/skills/example/install").status_code == 401
        assert client.get("/onboarding/brain-dump/status").status_code == 401

    install.assert_not_awaited()
    get_dump.assert_not_awaited()


@pytest.mark.parametrize("disabled_flag", ["HIRE_EXPERTS", "ONBOARDING_EXPERT_TEAM"])
async def test_onboarding_team_requires_both_mature_flags(disabled_flag, monkeypatch):
    assert await onboarding.service._expert_team_enabled("user-1") is True

    monkeypatch.setenv(f"FORCE_FLAG_{disabled_flag}", "false")
    assert await onboarding.service._expert_team_enabled("user-1") is False


@pytest.mark.parametrize("disabled", [False, True])
async def test_activity_generation_and_visibility_share_default(
    disabled, monkeypatch, mocker
):
    if disabled:
        monkeypatch.setenv("FORCE_FLAG_AI_AGENT_EXECUTION_SUMMARY", "false")
    llm = mocker.patch.object(
        activity, "get_openai_client", side_effect=AssertionError("Unexpected LLM call")
    )
    db_client = mocker.Mock()
    result = await activity.generate_activity_status_for_execution(
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        execution_stats=GraphExecutionStats(
            failure_reason=ExecutionFailureReason.ENTITLEMENT_REQUIRED
        ),
        db_client=db_client,
        user_id="user-1",
        execution_status=ExecutionStatus.FAILED,
    )
    assert result == (
        None
        if disabled
        else {
            "activity_status": activity.ENTITLEMENT_REQUIRED_SUMMARY,
            "correctness_score": 0.0,
        }
    )
    llm.assert_not_called()
    assert db_client.mock_calls == []

    execution = GraphExecutionMeta(
        id="exec-1",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=ExecutionStatus.FAILED,
        stats=GraphExecutionMeta.Stats(
            activity_status="Stored summary", correctness_score=0.0
        ),
    )
    single = await activity_gate.hide_activity_summary_if_disabled(execution, "user-1")
    listed = await activity_gate.hide_activity_summaries_if_disabled(
        [execution], "user-1"
    )
    assert single.stats is not None
    assert single.stats.activity_status == (None if disabled else "Stored summary")
    assert single.stats.correctness_score == (None if disabled else 0.0)
    assert listed == [single]


@pytest.mark.parametrize("disabled", [False, True])
async def test_home_persisted_activity_uses_same_default(disabled, monkeypatch, mocker):
    if disabled:
        monkeypatch.setenv("FORCE_FLAG_AI_AGENT_EXECUTION_SUMMARY", "false")
    now = datetime.now(UTC)
    content = BriefingContent(
        generated_at=now,
        timezone="UTC",
        zero_expert_fallback=True,
        run_items=[],
        decision_items=[],
        narrative="Completed today's work.",
    )
    mocker.patch.object(
        home.briefing_db,
        "get_briefing_for_date",
        return_value=SimpleNamespace(id="briefing-1", content=content.model_dump()),
    )

    visible = await home._persisted_briefing(
        user_id="user-1", timezone_name="UTC", now=now
    )

    assert visible is not None
    assert visible.narrative == (None if disabled else content.narrative)
