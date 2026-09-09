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
