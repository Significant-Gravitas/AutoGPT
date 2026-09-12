"""Tests for the copilot-skills routes."""

from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi.routing import APIRoute

from backend.api.features.skills.routes import router
from backend.api.features.store.exceptions import VirusDetectedError
from backend.api.rest_api import app as real_app
from backend.copilot.tools.skills import (
    BuiltInSkillError,
    ParsedSkill,
    SkillLimitError,
    SkillNotFoundError,
)

app = fastapi.FastAPI()
app.include_router(router, prefix="/skills")
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


EXPECTED_OPERATIONS = {
    ("get", "/api/skills"): "listCopilotSkills",
    ("post", "/api/skills"): "uploadCopilotSkill",
    ("delete", "/api/skills/{name}"): "deleteCopilotSkill",
    ("get", "/api/skills/{name}"): "readCopilotSkill",
}


@pytest.mark.parametrize(
    "method,path,operation_id",
    [(m, p, oid) for (m, p), oid in EXPECTED_OPERATIONS.items()],
)
def test_skill_operation_is_published(method: str, path: str, operation_id: str):
    """The mounted surface is contract: the generated frontend client is built from it."""
    operation = real_app.openapi()["paths"][path][method]
    assert operation["operationId"] == operation_id
    assert operation["tags"] == ["v1", "skills"]


def test_skill_surface_has_no_other_operations():
    published = {
        (method, path)
        for path, operations in real_app.openapi()["paths"].items()
        if path.startswith("/api/skills")
        for method in operations
    }
    assert published == set(EXPECTED_OPERATIONS)


@pytest.mark.parametrize("path", sorted({p for _, p in EXPECTED_OPERATIONS}))
def test_skill_path_is_served_by_this_module(path: str):
    handlers = {
        route.endpoint.__module__
        for route in real_app.routes
        if isinstance(route, APIRoute) and route.path == path
    }
    assert handlers == {"backend.api.features.skills.routes"}


def test_list_copilot_skills_returns_user_skills(
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /skills returns user-distilled skills (defaults are excluded
    because the UI hides them).
    """

    mocker.patch(
        "backend.api.features.skills.routes.list_user_skills",
        AsyncMock(
            return_value=[
                ParsedSkill(
                    name="oauth_flow",
                    description="OAuth handshake recipe",
                    body="...",
                    triggers=("auth", "oauth"),
                ),
                ParsedSkill(
                    name="zzz_cleanup",
                    description="Cleanup recipe",
                    body="...",
                ),
            ]
        ),
    )

    response = client.get("/skills")
    assert response.status_code == 200
    body = response.json()
    assert [s["name"] for s in body] == ["oauth_flow", "zzz_cleanup"]
    assert body[0]["triggers"] == ["auth", "oauth"]
    assert body[1]["triggers"] == []


def test_delete_copilot_skill_returns_name_on_success(
    mocker: pytest_mock.MockFixture,
) -> None:
    """DELETE /skills/{name} returns the slug and forwards the user_id."""
    delete_mock = AsyncMock(return_value="my_skill")
    mocker.patch("backend.api.features.skills.routes.delete_user_skill", delete_mock)

    response = client.delete("/skills/my_skill")
    assert response.status_code == 200
    assert response.json() == {"name": "my_skill"}
    delete_mock.assert_awaited_once()


def test_delete_copilot_skill_rejects_builtin(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Built-in defaults must not be user-deletable via the REST endpoint."""

    mocker.patch(
        "backend.api.features.skills.routes.delete_user_skill",
        AsyncMock(side_effect=BuiltInSkillError("built-in")),
    )

    response = client.delete("/skills/agent_building_guide")
    assert response.status_code == 400


def test_delete_copilot_skill_returns_404_when_missing(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Missing skills surface as 404 so the UI can reconcile its list."""

    mocker.patch(
        "backend.api.features.skills.routes.delete_user_skill",
        AsyncMock(side_effect=SkillNotFoundError("gone")),
    )

    response = client.delete("/skills/missing")
    assert response.status_code == 404


def test_read_copilot_skill_returns_user_body(
    mocker: pytest_mock.MockFixture,
) -> None:
    """GET /skills/{name} returns the full SKILL.md body for a user skill."""

    mocker.patch(
        "backend.api.features.skills.routes.read_user_skill_with_body",
        AsyncMock(
            return_value=ParsedSkill(
                name="oauth_flow",
                description="OAuth handshake recipe",
                body="# OAuth flow\n\nStep 1: ...",
                triggers=("auth",),
                version="1",
            )
        ),
    )

    response = client.get("/skills/oauth_flow")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "oauth_flow"
    assert body["body"].startswith("# OAuth flow")
    assert body["triggers"] == ["auth"]
    assert body["version"] == "1"
    assert body["is_default"] is False


def test_read_copilot_skill_returns_default_with_body(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A built-in default name returns is_default=True and a non-empty body."""

    mocker.patch(
        "backend.api.features.skills.routes.get_default_skill_with_body",
        return_value=ParsedSkill(
            name="agent_building_guide",
            description="default desc",
            body="# Default body\n",
            triggers=("create_agent",),
        ),
    )

    response = client.get("/skills/agent_building_guide")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "agent_building_guide"
    assert body["is_default"] is True
    assert body["body"].startswith("# Default body")


def test_read_copilot_skill_returns_404_when_missing(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A user-skill slug that has no SKILL.md surfaces as 404."""
    mocker.patch(
        "backend.api.features.skills.routes.read_user_skill_with_body",
        AsyncMock(return_value=None),
    )

    response = client.get("/skills/missing")
    assert response.status_code == 404


_VALID_SKILL_MD = (
    "---\n"
    "name: oauth_flow\n"
    "description: OAuth handshake recipe\n"
    "triggers:\n"
    "  - auth\n"
    "---\n\n"
    "# OAuth flow\n\nStep 1: redirect to /authorize\n"
)


def test_upload_copilot_skill_creates_skill(
    mocker: pytest_mock.MockFixture,
) -> None:
    """POST /skills parses the SKILL.md and persists it via store_user_skill."""
    store_mock = AsyncMock(
        return_value=ParsedSkill(
            name="oauth_flow",
            description="OAuth handshake recipe",
            body="# OAuth flow",
            triggers=("auth",),
        )
    )
    mocker.patch("backend.api.features.skills.routes.store_user_skill", store_mock)

    response = client.post("/skills", json={"content": _VALID_SKILL_MD})
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "oauth_flow"
    assert body["triggers"] == ["auth"]
    store_mock.assert_awaited_once()
    assert store_mock.await_args.kwargs["name"] == "oauth_flow"


def test_upload_copilot_skill_rejects_malformed_markdown() -> None:
    """A file without valid frontmatter returns 400 before touching storage."""
    response = client.post(
        "/skills", json={"content": "just some text, no frontmatter"}
    )
    assert response.status_code == 400


def test_upload_copilot_skill_returns_409_when_at_cap(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The per-user cap surfaces as 409 so the UI can prompt a delete."""
    mocker.patch(
        "backend.api.features.skills.routes.store_user_skill",
        AsyncMock(side_effect=SkillLimitError("Skill limit reached (50).")),
    )

    response = client.post("/skills", json={"content": _VALID_SKILL_MD})
    assert response.status_code == 409


def test_upload_copilot_skill_returns_400_on_validation_error(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A validation failure from store_user_skill maps to 400."""
    mocker.patch(
        "backend.api.features.skills.routes.store_user_skill",
        AsyncMock(side_effect=ValueError("name must be a slug")),
    )

    response = client.post("/skills", json={"content": _VALID_SKILL_MD})
    assert response.status_code == 400


def test_upload_copilot_skill_returns_400_on_virus_detection(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A virus-scan rejection surfaces as a 400 client error, not a 500."""
    mocker.patch(
        "backend.api.features.skills.routes.store_user_skill",
        AsyncMock(side_effect=VirusDetectedError("nasty")),
    )

    response = client.post("/skills", json={"content": _VALID_SKILL_MD})
    assert response.status_code == 400
    assert "virus scan" in response.json()["detail"]


_EXPERT_SKILL_ROUTES = {
    "list": lambda expert_id: client.get("/skills", params=_owner(expert_id)),
    "upload": lambda expert_id: client.post(
        "/skills", params=_owner(expert_id), json={"content": _VALID_SKILL_MD}
    ),
    "read": lambda expert_id: client.get(
        "/skills/oauth_flow", params=_owner(expert_id)
    ),
    "delete": lambda expert_id: client.delete(
        "/skills/oauth_flow", params=_owner(expert_id)
    ),
}

_SKILL_LAYER = {
    "list_user_skills": [],
    "store_user_skill": ParsedSkill(name="oauth_flow", description="d", body="b"),
    "read_user_skill_with_body": ParsedSkill(
        name="oauth_flow", description="d", body="b"
    ),
    "delete_user_skill": "oauth_flow",
}


@pytest.mark.parametrize("route", list(_EXPERT_SKILL_ROUTES))
def test_expert_skill_routes_refuse_an_expert_the_caller_does_not_own(
    route: str,
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """Every /skills route naming an expert goes through the same owner gate,
    and a refusal is a 404 that never reaches the skills layer."""
    owns = mocker.patch(
        "backend.api.features.skills.routes.experts_db.owns_private_active_expert",
        AsyncMock(return_value=False),
    )
    downstream = {
        name: mocker.patch(f"backend.api.features.skills.routes.{name}", AsyncMock())
        for name in _SKILL_LAYER
    }

    response = _EXPERT_SKILL_ROUTES[route]("expert-not-mine")

    assert response.status_code == 404
    assert response.json()["detail"] == "Expert 'expert-not-mine' not found"
    owns.assert_awaited_once_with(test_user_id, "expert-not-mine")
    for mock in downstream.values():
        mock.assert_not_awaited()


@pytest.mark.parametrize(
    "route,downstream",
    [
        ("list", "list_user_skills"),
        ("upload", "store_user_skill"),
        ("read", "read_user_skill_with_body"),
        ("delete", "delete_user_skill"),
    ],
)
def test_expert_skill_routes_forward_an_owned_expert(
    route: str,
    downstream: str,
    mocker: pytest_mock.MockFixture,
) -> None:
    """An owned PRIVATE expert passes the gate and its id reaches the skills
    layer, so the call reads that expert's folder and not Otto's."""
    mocker.patch(
        "backend.api.features.skills.routes.experts_db.owns_private_active_expert",
        AsyncMock(return_value=True),
    )
    mocker.patch(
        "backend.api.features.skills.routes.list_user_skill_sibling_paths",
        AsyncMock(return_value=[]),
    )
    target = mocker.patch(
        f"backend.api.features.skills.routes.{downstream}",
        AsyncMock(return_value=_SKILL_LAYER[downstream]),
    )

    response = _EXPERT_SKILL_ROUTES[route]("expert-mine")

    assert response.status_code in (200, 201)
    assert _forwarded_expert_id(target) == "expert-mine"


@pytest.mark.parametrize("route", list(_EXPERT_SKILL_ROUTES))
def test_personal_autopilot_skill_routes_skip_the_expert_gate(
    route: str,
    mocker: pytest_mock.MockFixture,
) -> None:
    """No expert named means personal Otto, which owns its own folder and
    must never be refused by the expert gate."""
    owns = mocker.patch(
        "backend.api.features.skills.routes.experts_db.owns_private_active_expert",
        AsyncMock(return_value=False),
    )
    mocker.patch(
        "backend.api.features.skills.routes.list_user_skill_sibling_paths",
        AsyncMock(return_value=[]),
    )
    for name, value in _SKILL_LAYER.items():
        mocker.patch(
            f"backend.api.features.skills.routes.{name}", AsyncMock(return_value=value)
        )

    response = _EXPERT_SKILL_ROUTES[route](None)

    assert response.status_code in (200, 201)
    owns.assert_not_awaited()


# ``expert_id`` omitted entirely, not sent empty: an empty query value is a
# str, which would take the expert branch of the gate rather than the
# personal-Otto one.
def _owner(expert_id: str | None) -> dict[str, str]:
    return {} if expert_id is None else {"expert_id": expert_id}


def _forwarded_expert_id(mock: AsyncMock) -> str | None:
    """``list_user_skills`` takes the owner positionally, the rest by keyword."""
    call = mock.await_args
    if call is None:
        return None
    if "expert_id" in call.kwargs:
        return call.kwargs["expert_id"]
    return call.args[1] if len(call.args) > 1 else None
