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
    ExpertIdentity,
    ExpertPod,
    ExpertRun,
    ExpertSoulUpdate,
    ExpertWorkflowRef,
    HireResult,
    RaiseAttachment,
    RaiseAttachmentFailure,
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


def _make_raised_expert(
    *,
    name: str,
    role: str = "",
    voice_preferences: str = "",
    workflows: list[ExpertWorkflowRef] | None = None,
    **overrides,
) -> Expert:
    return _make_expert(
        name=name,
        role=role,
        tagline=None,
        identity=experts_db._raised_identity(name),
        voice_preferences=voice_preferences,
        boundaries="",
        source_template_id=None,
        workflows=workflows or [],
        **overrides,
    )


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


def test_hire_expert_at_cap_returns_409(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.hire_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertLimitExceededError(20),
    )

    response = client.post("/experts", json={"template_id": "template-1"})

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "active_expert_limit",
        "limit": 20,
    }
    configured_snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True),
        "expert_hire_active_cap",
    )


# ─── Raise ─────────────────────────────────────────────────────────────


def test_create_raised_expert_returns_expert(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    raised = _make_raised_expert(id="raised-1", name="Otto")
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        return_value=RaiseResult(
            expert=raised,
            failed_attachments=[],
        ),
    )

    response = client.post("/experts/raise", json={"name": "Otto"})

    assert response.status_code == 200
    data = response.json()
    assert data["expert"]["id"] == "raised-1"
    assert data["expert"]["source_template_id"] is None
    assert data["failed_attachments"] == []
    mock_create.assert_awaited_once_with(
        test_user_id,
        "Otto",
        None,
        None,
        avatar_url=None,
        color=None,
        about=None,
        weekly_budget=None,
        attachments=[],
    )
    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True), "expert_raise_default"
    )


def test_create_raised_expert_passes_role_voice_budget_and_attachments(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        return_value=RaiseResult(
            expert=_make_raised_expert(
                id="raised-2",
                name="Nova",
                role="Research Assistant",
                voice_preferences="Warm and detailed.",
                workflows=[_make_workflow_ref()],
            ),
            failed_attachments=[],
        ),
    )

    response = client.post(
        "/experts/raise",
        json={
            "name": "Nova",
            "role": "Research Assistant",
            "voice_preferences": "Warm and detailed.",
            "weekly_budget": 250,
            "attachments": [
                {
                    "kind": "workflow",
                    "source": "marketplace",
                    "id": "listing-version-1",
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["failed_attachments"] == []
    mock_create.assert_awaited_once_with(
        test_user_id,
        "Nova",
        "Research Assistant",
        "Warm and detailed.",
        avatar_url=None,
        color=None,
        about=None,
        weekly_budget=250,
        attachments=[
            RaiseAttachment(
                kind="workflow", source="marketplace", id="listing-version-1"
            )
        ],
    )
    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True), "expert_raise_attachments"
    )


def test_create_raised_expert_forwards_about(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        return_value=RaiseResult(
            expert=_make_raised_expert(id="raised-4", name="Nova"),
            failed_attachments=[],
        ),
    )

    response = client.post(
        "/experts/raise",
        json={"name": "Nova", "about": "  Always cites a source.  "},
    )

    assert response.status_code == 200
    mock_create.assert_awaited_once_with(
        test_user_id,
        "Nova",
        None,
        None,
        avatar_url=None,
        color=None,
        about="Always cites a source.",
        weekly_budget=None,
        attachments=[],
    )


def test_create_raised_expert_reports_attachment_installation_failure(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        return_value=RaiseResult(
            expert=_make_raised_expert(id="raised-3", name="Nova"),
            failed_attachments=[
                RaiseAttachmentFailure(
                    kind="workflow",
                    source="marketplace",
                    id="listing-version-1",
                    reason="installation_failed",
                )
            ],
        ),
    )

    response = client.post(
        "/experts/raise",
        json={
            "name": "Nova",
            "attachments": [
                {
                    "kind": "workflow",
                    "source": "marketplace",
                    "id": "listing-version-1",
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["failed_attachments"][0]["reason"] == "installation_failed"
    mock_create.assert_awaited_once_with(
        test_user_id,
        "Nova",
        None,
        None,
        avatar_url=None,
        color=None,
        about=None,
        weekly_budget=None,
        attachments=[
            RaiseAttachment(
                kind="workflow", source="marketplace", id="listing-version-1"
            )
        ],
    )
    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True),
        "expert_raise_attachment_installation_failure",
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


def test_create_raised_expert_passes_avatar_and_color(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        return_value=RaiseResult(
            expert=_make_raised_expert(id="raised-3", name="Nova"),
            failed_attachments=[],
        ),
    )

    response = client.post(
        "/experts/raise",
        json={
            "name": "Nova",
            "avatar_url": "  https://storage.googleapis.com/bucket/nova.png  ",
            "color": "  sky-300  ",
        },
    )

    assert response.status_code == 200
    mock_create.assert_awaited_once_with(
        test_user_id,
        "Nova",
        None,
        None,
        avatar_url="https://storage.googleapis.com/bucket/nova.png",
        color="sky-300",
        about=None,
        weekly_budget=None,
        attachments=[],
    )


def test_create_raised_expert_accepts_relative_avatar_path(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        return_value=RaiseResult(
            expert=_make_raised_expert(id="raised-4", name="Otto"),
            failed_attachments=[],
        ),
    )

    response = client.post(
        "/experts/raise",
        json={"name": "Otto", "avatar_url": "/experts/maria.svg"},
    )

    assert response.status_code == 200
    assert mock_create.await_args.kwargs["avatar_url"] == "/experts/maria.svg"


@pytest.mark.parametrize(
    "avatar_url",
    [
        "javascript:alert(1)",
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "//evil.example.com/x.png",
        "ftp://example.com/x.png",
    ],
)
def test_create_raised_expert_rejects_unsafe_avatar_url(
    mocker: pytest_mock.MockerFixture,
    avatar_url: str,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/experts/raise",
        json={"name": "Otto", "avatar_url": avatar_url},
    )

    assert response.status_code == 422
    mock_create.assert_not_awaited()


def test_create_raised_expert_treats_blank_avatar_and_color_as_unset(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        return_value=RaiseResult(
            expert=_make_raised_expert(id="raised-5", name="Otto"),
            failed_attachments=[],
        ),
    )

    response = client.post(
        "/experts/raise",
        json={"name": "Otto", "avatar_url": "   ", "color": "   "},
    )

    assert response.status_code == 200
    assert mock_create.await_args.kwargs == {
        "avatar_url": None,
        "color": None,
        "about": None,
        "weekly_budget": None,
        "attachments": [],
    }


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
        ("weekly_budget", -1, "expert_raise_weekly_budget_negative"),
        ("color", "c" * 33, "expert_raise_color_too_long"),
        ("about", "a" * 10_001, "expert_raise_about_too_long"),
        (
            "avatar_url",
            f"https://cdn.example.com/{'a' * 2_001}.png",
            "expert_raise_avatar_url_too_long",
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
    configured_snapshot: Snapshot,
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
    configured_snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True),
        "expert_raise_active_cap",
    )


def test_create_raised_expert_at_lifetime_cap_returns_409(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
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
    configured_snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True),
        "expert_raise_lifetime_cap",
    )


def test_create_raised_expert_unavailable_attachment_returns_404(
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_raised_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.FirstJobUnavailableError(
            "workflow", "marketplace", "listing-version-9"
        ),
    )

    response = client.post(
        "/experts/raise",
        json={
            "name": "Otto",
            "attachments": [
                {
                    "kind": "workflow",
                    "source": "marketplace",
                    "id": "listing-version-9",
                }
            ],
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == {
        "code": "attachment_unavailable",
        "kind": "workflow",
        "source": "marketplace",
        "id": "listing-version-9",
    }
    configured_snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True),
        "expert_raise_attachment_unavailable",
    )


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


def test_list_expert_identities_returns_lifetime_roster_projection(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    identities = [
        ExpertIdentity(
            id="expert-1",
            name="Maria",
            avatar_url=None,
            role="Marketing Specialist",
            is_archived=True,
        )
    ]
    mock_list = mocker.patch(
        "backend.api.features.experts.routes.experts_db.list_expert_identities",
        new_callable=AsyncMock,
        return_value=identities,
    )

    response = client.get("/experts/identities")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "expert-1",
            "name": "Maria",
            "avatar_url": None,
            "role": "Marketing Specialist",
            "is_archived": True,
        }
    ]
    mock_list.assert_awaited_once_with(test_user_id)


# ─── Runs ──────────────────────────────────────────────────────────────


def _make_run(**overrides) -> ExpertRun:
    values = {
        "execution_id": "exec-1",
        "graph_id": "graph-1",
        "agent_name": "SEO Blog Writer",
        "library_agent_id": "library-agent-1",
        "status": "completed",
        "output_type": "table",
        "output_key": "result",
        "needs_review": False,
        "started_at": None,
        "ended_at": None,
    }
    values.update(overrides)
    values["link"] = overrides.get(
        "link",
        "/library/agents/"
        f"{values['library_agent_id']}?activeTab=runs&activeItem="
        f"{values['execution_id']}",
    )
    return ExpertRun(**values)


def test_list_expert_runs_forwards_authenticated_user_and_serializes(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    runs = [_make_run(), _make_run(execution_id="exec-2", output_type="doc")]
    mock_list = mocker.patch(
        "backend.api.features.experts.routes.experts_db.list_expert_runs",
        new_callable=AsyncMock,
        return_value=runs,
    )

    response = client.get("/experts/expert-1/runs")

    assert response.status_code == 200
    data = response.json()
    assert [r["execution_id"] for r in data] == ["exec-1", "exec-2"]
    assert data[0]["output_type"] == "table"
    mock_list.assert_awaited_once_with(test_user_id, "expert-1")

    configured_snapshot.assert_match(
        f"{json.dumps(data, indent=2, sort_keys=True)}\n", "expert_runs_list"
    )


def test_list_expert_runs_unknown_expert_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.list_expert_runs",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertNotFoundError("nope"),
    )

    response = client.get("/experts/nope/runs")

    assert response.status_code == 404


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


def test_hire_existing_shared_expert_returns_generic_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.hire_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertNotFoundError("shared-expert"),
    )

    response = client.post("/experts", json={"template_id": "template-1"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Expert not found"}


def test_rehire_expert_workspace_unavailable_returns_503(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.hire_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertPrivateTenancyNotFoundError("expert-1"),
    )

    response = client.post("/experts", json={"template_id": "template-1"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Your expert workspace is still being set up. Try again shortly."
    }


def test_rehire_dependency_unavailable_returns_503(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.hire_expert",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertHireUnavailableError("expert-1"),
    )

    response = client.post("/experts", json={"template_id": "template-1"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Your expert is temporarily unavailable. Try again shortly."
    }


# ─── Pods ──────────────────────────────────────────────────────────────


def _make_pod(**overrides) -> ExpertPod:
    values = {
        "id": "pod-1",
        "name": "Growth",
        "created_at": "2026-08-14T00:00:00Z",
    }
    values.update(overrides)
    return ExpertPod(**values)


def test_create_pod_returns_pod(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_pod",
        new_callable=AsyncMock,
        return_value=_make_pod(),
    )

    response = client.post("/experts/pods", json={"name": "Growth"})

    assert response.status_code == 200
    assert response.json()["name"] == "Growth"
    mock_create.assert_awaited_once_with(test_user_id, "Growth")
    configured_snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True), "expert_pod_create"
    )


def test_create_pod_strips_name(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_pod",
        new_callable=AsyncMock,
        return_value=_make_pod(),
    )

    response = client.post("/experts/pods", json={"name": "  Growth  "})

    assert response.status_code == 200
    mock_create.assert_awaited_once_with(test_user_id, "Growth")


@pytest.mark.parametrize("name", ["", "   "])
def test_create_pod_rejects_blank_name(
    name: str,
    mocker: pytest_mock.MockerFixture,
    configured_snapshot: Snapshot,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_pod",
        new_callable=AsyncMock,
        return_value=_make_pod(),
    )

    response = client.post("/experts/pods", json={"name": name})

    assert response.status_code == 422
    mock_create.assert_not_awaited()
    if name == "   ":
        configured_snapshot.assert_match(
            json.dumps(response.json(), indent=2, sort_keys=True),
            "expert_pod_create_blank_name",
        )


def test_create_pod_rejects_name_over_max_length(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_create = mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_pod",
        new_callable=AsyncMock,
        return_value=_make_pod(),
    )

    response = client.post("/experts/pods", json={"name": "x" * 101})

    assert response.status_code == 422
    mock_create.assert_not_awaited()


def test_create_pod_at_limit_returns_409(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_pod",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertPodLimitReachedError(experts_db.MAX_PODS_PER_USER),
    )

    response = client.post("/experts/pods", json={"name": "Growth"})

    assert response.status_code == 409
    assert str(experts_db.MAX_PODS_PER_USER) in response.json()["detail"]


def test_create_pod_duplicate_name_returns_409(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.create_pod",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertPodNameTakenError("Growth"),
    )

    response = client.post("/experts/pods", json={"name": "Growth"})

    assert response.status_code == 409


def test_list_pods_returns_user_pods(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    mock_list = mocker.patch(
        "backend.api.features.experts.routes.experts_db.list_pods",
        new_callable=AsyncMock,
        return_value=[_make_pod()],
    )

    response = client.get("/experts/pods")

    assert response.status_code == 200
    data = response.json()
    assert data[0]["id"] == "pod-1"
    # Membership is not embedded — clients group from Expert.pod_id, so the
    # listing must not carry a members payload at all.
    assert "members" not in data[0]
    mock_list.assert_awaited_once_with(test_user_id)
    configured_snapshot.assert_match(
        json.dumps(data, indent=2, sort_keys=True), "expert_pods_list"
    )


def test_pods_route_is_not_shadowed_by_expert_detail(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """ "/pods" must win over "/{expert_id}", which only holds while the pod
    routes stay declared first. Fails loudly if they are ever reordered."""
    mock_list = mocker.patch(
        "backend.api.features.experts.routes.experts_db.list_pods",
        new_callable=AsyncMock,
        return_value=[_make_pod()],
    )
    mock_get = mocker.patch(
        "backend.api.features.experts.routes.experts_db.get_expert",
        new_callable=AsyncMock,
        return_value=_make_expert(),
    )

    response = client.get("/experts/pods")

    assert response.status_code == 200
    mock_list.assert_awaited_once()
    mock_get.assert_not_awaited()


def test_assign_pod_returns_updated_expert(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    configured_snapshot: Snapshot,
) -> None:
    mock_assign = mocker.patch(
        "backend.api.features.experts.routes.experts_db.assign_pod",
        new_callable=AsyncMock,
        return_value=_make_expert(pod_id="pod-1"),
    )

    response = client.patch("/experts/expert-1/pod", json={"pod_id": "pod-1"})

    assert response.status_code == 200
    assert response.json()["pod_id"] == "pod-1"
    mock_assign.assert_awaited_once_with(test_user_id, "expert-1", "pod-1")
    configured_snapshot.assert_match(
        json.dumps(response.json(), indent=2, sort_keys=True), "expert_pod_assign"
    )


def test_assign_pod_null_detaches(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_assign = mocker.patch(
        "backend.api.features.experts.routes.experts_db.assign_pod",
        new_callable=AsyncMock,
        return_value=_make_expert(pod_id=None),
    )

    response = client.patch("/experts/expert-1/pod", json={"pod_id": None})

    assert response.status_code == 200
    assert response.json()["pod_id"] is None
    mock_assign.assert_awaited_once_with(test_user_id, "expert-1", None)


def test_assign_pod_requires_explicit_pod_id(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """An empty body must not be interpreted as a detach."""
    mock_assign = mocker.patch(
        "backend.api.features.experts.routes.experts_db.assign_pod",
        new_callable=AsyncMock,
        return_value=_make_expert(pod_id=None),
    )

    response = client.patch("/experts/expert-1/pod", json={})

    assert response.status_code == 422
    mock_assign.assert_not_awaited()


def test_assign_pod_unknown_expert_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.assign_pod",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertNotFoundError("nope"),
    )

    response = client.patch("/experts/nope/pod", json={"pod_id": "pod-1"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Expert or pod not found"


def test_assign_pod_unknown_pod_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.experts.routes.experts_db.assign_pod",
        new_callable=AsyncMock,
        side_effect=experts_db.ExpertPodNotFoundError("nope"),
    )

    response = client.patch("/experts/expert-1/pod", json={"pod_id": "nope"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Expert or pod not found"
