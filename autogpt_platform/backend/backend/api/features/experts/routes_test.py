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
from backend.api.features.experts.models import Expert, ExpertWorkflowRef, HireResult
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
