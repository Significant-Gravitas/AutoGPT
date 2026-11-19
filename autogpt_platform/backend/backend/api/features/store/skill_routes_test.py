"""Public skill browsing and authenticated owner-scoped installation."""

import fastapi
import fastapi.testclient
import pytest

from backend.util.models import Pagination

from . import skill_model, skill_routes, skill_submission_db

app = fastapi.FastAPI()
app.include_router(skill_routes.router, prefix="/api/store/skills")
client = fastapi.testclient.TestClient(app)


@pytest.fixture
def authenticated(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_install_requires_authentication(
    monkeypatch: pytest.MonkeyPatch,
):
    """Public browsing does not authorize installation."""

    response = client.post("/api/store/skills/any-slug/install")

    assert response.status_code == 401


def test_submissions_is_a_route_of_its_own_not_a_skill_slug(
    monkeypatch: pytest.MonkeyPatch, mocker, authenticated
):
    """`/{slug}` is declared after `/submissions`, so the placeholder cannot
    swallow it — FastAPI matches in declaration order."""
    listed = mocker.patch.object(
        skill_submission_db, "list_my_skill_submissions", return_value=[]
    )
    detail = mocker.patch.object(skill_routes.skill_db, "get_marketplace_skill")

    response = client.get("/api/store/skills/submissions")

    assert response.status_code == 200
    listed.assert_awaited_once()
    detail.assert_not_called()


@pytest.mark.parametrize("expert_id", [None, "expert-1"], ids=["library", "expert"])
def test_install_targets_the_expert_the_caller_named(
    monkeypatch: pytest.MonkeyPatch, mocker, authenticated, test_user_id, expert_id
):
    mocker.patch.object(
        skill_routes.experts_db, "owns_private_active_expert", return_value=True
    )
    install = mocker.patch.object(
        skill_routes.skill_db,
        "install_marketplace_skill",
        return_value=skill_model.InstalledSkill(name="a-skill", required_providers=[]),
    )

    response = client.post(
        "/api/store/skills/a-skill/install",
        params={"expert_id": expert_id} if expert_id else {},
    )

    assert response.status_code == 200
    install.assert_awaited_once_with(test_user_id, "a-skill", expert_id=expert_id)


def test_install_onto_someone_elses_expert_is_a_404_that_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, mocker, authenticated, test_user_id
):
    """A 403 would confirm the expert exists, so an unowned id reads as missing."""
    owns = mocker.patch.object(
        skill_routes.experts_db, "owns_private_active_expert", return_value=False
    )
    install = mocker.patch.object(skill_routes.skill_db, "install_marketplace_skill")

    response = client.post(
        "/api/store/skills/a-skill/install", params={"expert_id": "not-mine"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Expert 'not-mine' not found"
    owns.assert_awaited_once_with(test_user_id, "not-mine")
    install.assert_not_awaited()


def test_public_skills_remain_available_with_a_retired_false_override(
    monkeypatch, mocker
):
    monkeypatch.setenv("FORCE_FLAG_SKILLS_HUB", "false")
    listed = mocker.patch.object(
        skill_routes.skill_db,
        "get_marketplace_skills",
        return_value=skill_model.MarketplaceSkillsResponse(
            skills=[],
            pagination=Pagination(
                total_items=0, total_pages=0, current_page=1, page_size=20
            ),
        ),
    )
    response = client.get("/api/store/skills")
    assert response.status_code == 200
    listed.assert_awaited_once()


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/store/skills/any-slug/install"),
        ("post", "/api/store/skills/submissions"),
        ("get", "/api/store/skills/submissions"),
        ("put", "/api/store/skills/submissions/any-version"),
    ],
)
def test_private_skill_operations_require_auth_after_flag_retirement(method, path):
    assert client.request(method, path).status_code == 401
