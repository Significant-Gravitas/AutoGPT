import io
import zipfile
from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi.testclient import TestClient

from backend.api.features.experts import package_routes
from backend.api.features.experts.package_model import (
    ExpertManifest,
    ExpertPackage,
    ExpertPackageError,
)
from backend.util.exceptions import NotFoundError

URL = "/templates/tmpl-1/package"


def _package() -> ExpertPackage:
    manifest = ExpertManifest(
        name="Ada Lovelace",
        role="Research assistant",
        job_title=None,
        tagline=None,
        bio=None,
        categories=[],
        color="",
        avatar_url=None,
        identity="You are Ada.",
        voice_preferences="",
        voice_samples=[],
        boundaries="",
        day_one=[],
        skills=[],
        workflows=[],
        routines=[],
    )
    return ExpertPackage(manifest=manifest, skills={}, workflows={})


@pytest.fixture
def app() -> fastapi.FastAPI:
    app = fastapi.FastAPI()
    app.include_router(package_routes.router)
    return app


@pytest.fixture
def client(app: fastapi.FastAPI, mock_jwt_user) -> TestClient:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    return TestClient(app)


def test_download_requires_authentication(app: fastapi.FastAPI):
    assert TestClient(app).get(URL).status_code == 401


def test_download_returns_a_named_zip_for_the_caller(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, mock_jwt_user
):
    build = AsyncMock(return_value=_package())
    monkeypatch.setattr(package_routes, "build_template_package", build)
    response = client.get(URL)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="ada-lovelace.expert.zip"'
    )
    assert zipfile.ZipFile(io.BytesIO(response.content)).namelist() == ["expert.json"]
    build.assert_awaited_once_with("tmpl-1", mock_jwt_user["user_id"])


def test_unknown_template_is_404(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        package_routes,
        "build_template_package",
        AsyncMock(side_effect=NotFoundError("Expert template #tmpl-1 not found")),
    )
    assert client.get(URL).status_code == 404


def test_oversized_package_is_413(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        package_routes, "build_template_package", AsyncMock(return_value=_package())
    )
    monkeypatch.setattr(
        package_routes,
        "zip_from_expert_package",
        MagicMock(side_effect=ExpertPackageError("too big", over_limit=True)),
    )
    assert client.get(URL).status_code == 413


def test_malformed_package_is_400(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        package_routes, "build_template_package", AsyncMock(return_value=_package())
    )
    monkeypatch.setattr(
        package_routes,
        "zip_from_expert_package",
        MagicMock(side_effect=ExpertPackageError("skills differ")),
    )
    assert client.get(URL).status_code == 400
