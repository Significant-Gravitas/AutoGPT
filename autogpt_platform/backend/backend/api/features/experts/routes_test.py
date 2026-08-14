"""Tests for the experts API routes.

Pattern mirrors backend/api/features/library/routes_test.py: a local FastAPI
app, the global `mock_jwt_user` auth override fixture, and `experts_db` mocked
with AsyncMock at the route module's import site.
"""

import json
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from pytest_snapshot.plugin import Snapshot

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    Expert,
    ExpertSoulUpdate,
    ExpertWorkflowRef,
    HireResult,
    RaiseResult,
)
from backend.api.features.experts.routes import router

app = fastapi.FastAPI()
app.include_router(router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


def _make_expert(**overrides) -> Expert:
    values = {
        "id": "expert-1",
        "name": "Maria",
        "avatar_url": None,
        "role": "Marketing Specialist",
        "tagline": "Grows your audience",
        "bio": None,
        "skills": [],
        "identity": "You are Maria, a pragmatic marketing specialist.",
        "voice_preferences": "Direct and concise.",
        "boundaries": "Ask before external actions.",
        "protected_soul_rules": list(PROTECTED_SOUL_RULES),
        "is_template": False,
        "source_template_id": "template-1",
        "is_archived": False,
        "workflows": [],
    }
    values.update(overrides)
    return Expert(**values)


def _make_workflow_ref(**overrides) -> ExpertWorkflowRef:
    values = {
        "id": "workflow-ref-1",
        "store_listing_version_id": "listing-version-1",
        "library_agent_id": "library-agent-1",
        "graph_id": "graph-1",
        "name": "SEO Blog Writer",
        "description": "Writes SEO-optimized blog posts",
    }
    values.update(overrides)
    return ExpertWorkflowRef(**values)


# ─── List templates ────────────────────────────────────────────────────


def test_list_expert_templates(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
) -> None:
    template = _make_expert(
        id="template-1",
        is_template=True,
        source_template_id=None,
        workflows=[
            _make_workflow_ref(library_agent_id=None, graph_id=None),
        ],
    )
    mock_list = mocker.patch(
        "backend.api.features.experts.routes.experts_db.list_templates",
        new_callable=AsyncMock,
        return_value=[template],
    )

    response = client.get("/experts/templates")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == "template-1"
    assert data[0]["is_template"] is True
    mock_list.assert_awaited_once_with()

    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True), "expert_templates_list"
    )


# ─── Hire ──────────────────────────────────────────────────────────────


def test_hire_expert_returns_expert_and_empty_failed_preloads(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    hired = _make_expert()
    mock_hire = mocker.patch(
        "backend.api.features.experts.routes.experts_db.hire_expert",
        new_callable=AsyncMock,
        return_value=HireResult(expert=hired, failed_preloads=[]),
    )

    response = client.post("/experts", json={"template_id": "template-1"})

    assert response.status_code == 200
    data = response.json()
    assert data["expert"]["id"] == "expert-1"
    assert data["failed_preloads"] == []
    mock_hire.assert_awaited_once_with(test_user_id, "template-1", None)


def test_hire_expert_twice_returns_same_expert_id(
    mocker: pytest_mock.MockerFixture,
) -> None:
    hired = _make_expert()
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.hire_expert",
        new_callable=AsyncMock,
        return_value=HireResult(expert=hired, failed_preloads=[]),
    )

    first = client.post("/experts", json={"template_id": "template-1"})
    second = client.post("/experts", json={"template_id": "template-1"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["expert"]["id"] == second.json()["expert"]["id"]


def test_hire_expert_unknown_template_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.hire_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertTemplateNotFoundError("nope"),
    )

    response = client.post("/experts", json={"template_id": "nope"})

    assert response.status_code == 404


# ─── Raise ─────────────────────────────────────────────────────────────


def test_create_raised_expert_returns_expert(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    raised = _make_expert(id="raised-1", name="Otto", source_template_id=None)
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        return_value=RaiseResult(
            expert=raised,
            first_job_installed=False,
            first_job_failure_reason=None,
        ),
    )

    response = client.post("/experts/raise", json={"name": "Otto"})

    assert response.status_code == 200
    data = response.json()
    assert data["expert"]["id"] == "raised-1"
    assert data["expert"]["source_template_id"] is None
    assert data["first_job_installed"] is False
    assert data["first_job_failure_reason"] is None
    mock_create.assert_awaited_once_with(test_user_id, "Otto", None, None, None)
    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True), "expert_raise_default"
    )


def test_create_raised_expert_passes_role_voice_and_first_job(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        return_value=RaiseResult(
            expert=_make_expert(id="raised-2", source_template_id=None),
            first_job_installed=True,
            first_job_failure_reason=None,
        ),
    )

    response = client.post(
        "/experts/raise",
        json={
            "name": "Nova",
            "role": "Research Assistant",
            "voice_preferences": "Warm and detailed.",
            "first_job_store_listing_version_id": "listing-version-1",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["first_job_installed"] is True
    mock_create.assert_awaited_once_with(
        test_user_id,
        "Nova",
        "Research Assistant",
        "Warm and detailed.",
        "listing-version-1",
    )
    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True), "expert_raise_first_job"
    )


def test_create_raised_expert_requires_name(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
    )

    response = client.post("/experts/raise", json={"name": "   "})

    assert response.status_code == 422
    mock_create.assert_not_awaited()
    configured_snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True),
        "expert_raise_blank_name",
    )


@pytest.mark.parametrize(
    ("field", "value", "snapshot_name"),
    [
        ("name", "n" * 101, "expert_raise_name_too_long"),
        ("role", "r" * 101, "expert_raise_role_too_long"),
        (
            "voice_preferences",
            "v" * 4_001,
            "expert_raise_voice_preferences_too_long",
        ),
    ],
)
def test_create_raised_expert_rejects_overlong_fields(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
    field: str,
    value: str,
    snapshot_name: str,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
    )
    payload = {"name": "Otto", field: value}

    response = client.post("/experts/raise", json=payload)

    assert response.status_code == 422
    mock_create.assert_not_awaited()
    configured_snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True), snapshot_name
    )


def test_create_raised_expert_at_cap_returns_409(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertLimitExceededError(20),
    )

    response = client.post("/experts/raise", json={"name": "Otto"})

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "active_expert_limit",
        "limit": 20,
    }


def test_create_raised_expert_at_lifetime_cap_returns_409(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.RaisedExpertLifetimeLimitExceededError(100),
    )

    response = client.post("/experts/raise", json={"name": "Otto"})

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "raised_expert_lifetime_limit",
        "limit": 100,
    }


def test_create_raised_expert_unavailable_first_job_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.FirstJobUnavailableError("listing-version-9"),
    )

    response = client.post(
        "/experts/raise",
        json={
            "name": "Otto",
            "first_job_store_listing_version_id": "listing-version-9",
        },
    )

    assert response.status_code == 404


# ─── Get ───────────────────────────────────────────────────────────────


def test_get_expert_of_other_user_returns_404(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_get = mocker.patch(
        "backend.api.features.experts.routes.experts_db.get_expert",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.get("/experts/expert-1")

    assert response.status_code == 404
    mock_get.assert_awaited_once_with(test_user_id, "expert-1")


def test_get_expert_returns_expert(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_get = mocker.patch(
        "backend.api.features.experts.routes.experts_db.get_expert",
        new_callable=AsyncMock,
        return_value=_make_expert(),
    )

    response = client.get("/experts/expert-1")

    assert response.status_code == 200
    assert response.json()["id"] == "expert-1"
    mock_get.assert_awaited_once_with(test_user_id, "expert-1")


# ─── Soul ──────────────────────────────────────────────────────────────


def test_update_expert_soul_returns_updated_expert(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    updated = _make_expert(
        name="Mara",
        identity="You are Mara, a thoughtful strategist.",
        voice_preferences="Warm, concise, and direct.",
        boundaries="Never invent customer evidence.",
    )
    mock_update = mocker.patch(
        "backend.api.features.experts.routes.experts_db.update_soul",
        new_callable=AsyncMock,
        return_value=updated,
    )
    soul = {
        "name": "Mara",
        "identity": "You are Mara, a thoughtful strategist.",
        "voice_preferences": "Warm, concise, and direct.",
        "boundaries": "Never invent customer evidence.",
    }

    response = client.patch("/experts/expert-1/soul", json=soul)

    assert response.status_code == 200
    assert response.json()["name"] == "Mara"
    configured_snapshot.assert_match(
        f"{json.dumps(response.json(), indent=2, sort_keys=True)}\n",
        "expert_soul_update",
    )
    mock_update.assert_awaited_once_with(
        test_user_id, "expert-1", ExpertSoulUpdate(**soul)
    )


def test_update_expert_soul_not_found_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_update = mocker.patch(
        "backend.api.features.experts.routes.experts_db.update_soul",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertNotFoundError("expert-1"),
    )

    response = client.patch(
        "/experts/expert-1/soul",
        json={
            "name": "Mara",
            "identity": "You are Mara.",
            "voice_preferences": "Direct.",
            "boundaries": "Ask before sending.",
        },
    )

    assert response.status_code == 404
    mock_update.assert_awaited_once()


def test_update_expert_soul_validates_field_lengths() -> None:
    response = client.patch(
        "/experts/expert-1/soul",
        json={
            "name": "x" * 101,
            "identity": "You are Maria.",
            "voice_preferences": "Direct.",
            "boundaries": "Ask before sending.",
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["name", "identity"])
def test_update_expert_soul_rejects_blank_required_fields(
    field: str,
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_update = mocker.patch(
        "backend.api.features.experts.routes.experts_db.update_soul",
        new_callable=AsyncMock,
        return_value=_make_expert(),
    )
    soul = {
        "name": "Mara",
        "identity": "You are Mara.",
        "voice_preferences": "Direct.",
        "boundaries": "Ask before sending.",
    }
    soul[field] = "   "

    response = client.patch("/experts/expert-1/soul", json=soul)

    assert response.status_code == 422
    mock_update.assert_not_awaited()


def test_update_expert_soul_strips_required_fields(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_update = mocker.patch(
        "backend.api.features.experts.routes.experts_db.update_soul",
        new_callable=AsyncMock,
        return_value=_make_expert(),
    )

    response = client.patch(
        "/experts/expert-1/soul",
        json={
            "name": "  Mara  ",
            "identity": "  You are Mara.  ",
            "voice_preferences": "Direct.",
            "boundaries": "Ask before sending.",
        },
    )

    assert response.status_code == 200
    soul = mock_update.await_args.args[2]
    assert soul.name == "Mara"
    assert soul.identity == "You are Mara."


# ─── Install workflow ──────────────────────────────────────────────────


def test_install_workflow_duplicate_returns_same_row_id(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    ref = _make_workflow_ref()
    mock_install = mocker.patch(
        "backend.api.features.experts.routes.experts_db.install_workflow",
        new_callable=AsyncMock,
        return_value=ref,
    )

    body = {"store_listing_version_id": "listing-version-1"}
    first = client.post("/experts/expert-1/workflows", json=body)
    second = client.post("/experts/expert-1/workflows", json=body)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"] == "workflow-ref-1"
    assert mock_install.await_args_list == [
        mocker.call(test_user_id, "expert-1", "listing-version-1"),
        mocker.call(test_user_id, "expert-1", "listing-version-1"),
    ]


def test_install_workflow_unknown_expert_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.install_workflow",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertNotFoundError("nope"),
    )

    response = client.post(
        "/experts/nope/workflows", json={"store_listing_version_id": "listing-1"}
    )

    assert response.status_code == 404


# ─── Archive + list ────────────────────────────────────────────────────


def test_delete_then_list_excludes_archived(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    experts = [_make_expert(id="expert-1"), _make_expert(id="expert-2", name="Max")]

    async def _list_experts(user_id: str) -> list[Expert]:
        assert user_id == test_user_id
        return [e for e in experts if not e.is_archived]

    async def _archive_expert(user_id: str, expert_id: str) -> None:
        assert user_id == test_user_id
        for i, expert in enumerate(experts):
            if expert.id == expert_id:
                experts[i] = expert.model_copy(update={"is_archived": True})
                return
        raise experts_db.ExpertNotFoundError(expert_id)

    mocker.patch(
        "backend.api.features.experts.routes.experts_db.list_experts",
        new_callable=AsyncMock,
        side_effect=_list_experts,
    )
    mock_archive = mocker.patch(
        "backend.api.features.experts.routes.experts_db.archive_expert",
        new_callable=AsyncMock,
        side_effect=_archive_expert,
    )

    before = client.get("/experts")
    assert before.status_code == 200
    assert {e["id"] for e in before.json()} == {"expert-1", "expert-2"}

    delete_response = client.delete("/experts/expert-1")
    assert delete_response.status_code == 204
    mock_archive.assert_awaited_once_with(test_user_id, "expert-1")

    after = client.get("/experts")
    assert after.status_code == 200
    assert {e["id"] for e in after.json()} == {"expert-2"}


def test_delete_unknown_expert_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.archive_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertNotFoundError("nope"),
    )

    response = client.delete("/experts/nope")

    assert response.status_code == 404
