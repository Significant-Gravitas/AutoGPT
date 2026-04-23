"""Route tests: domain exceptions → HTTPException status codes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.util.exceptions import (
    LinkAlreadyExistsError,
    LinkFlowMismatchError,
    LinkTokenExpiredError,
    NotAuthorizedError,
    NotFoundError,
)


def _db_mock(**method_configs):
    """Return a mock of the accessor's return value with the given AsyncMocks."""
    db = MagicMock()
    for name, mock in method_configs.items():
        setattr(db, name, mock)
    return db


class TestTokenInfoRouteTranslation:
    @pytest.mark.asyncio
    async def test_not_found_maps_to_404(self):
        from backend.api.features.platform_linking.routes import (
            get_link_token_info_route,
        )

        db = _db_mock(
            get_link_token_info=AsyncMock(side_effect=NotFoundError("missing"))
        )
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            with pytest.raises(HTTPException) as exc:
                await get_link_token_info_route(token="abc")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_expired_maps_to_410(self):
        from backend.api.features.platform_linking.routes import (
            get_link_token_info_route,
        )

        db = _db_mock(
            get_link_token_info=AsyncMock(side_effect=LinkTokenExpiredError("expired"))
        )
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            with pytest.raises(HTTPException) as exc:
                await get_link_token_info_route(token="abc")
        assert exc.value.status_code == 410


class TestConfirmLinkRouteTranslation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc,expected_status",
        [
            (NotFoundError("missing"), 404),
            (LinkFlowMismatchError("wrong flow"), 400),
            (LinkTokenExpiredError("expired"), 410),
            (LinkAlreadyExistsError("already"), 409),
        ],
    )
    async def test_translation(self, exc: Exception, expected_status: int):
        from backend.api.features.platform_linking.routes import confirm_link_token

        db = _db_mock(confirm_server_link=AsyncMock(side_effect=exc))
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            with pytest.raises(HTTPException) as ctx:
                await confirm_link_token(token="abc", user_id="u1")
        assert ctx.value.status_code == expected_status


class TestConfirmUserLinkRouteTranslation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc,expected_status",
        [
            (NotFoundError("missing"), 404),
            (LinkFlowMismatchError("wrong flow"), 400),
            (LinkTokenExpiredError("expired"), 410),
            (LinkAlreadyExistsError("already"), 409),
        ],
    )
    async def test_translation(self, exc: Exception, expected_status: int):
        from backend.api.features.platform_linking.routes import confirm_user_link_token

        db = _db_mock(confirm_user_link=AsyncMock(side_effect=exc))
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            with pytest.raises(HTTPException) as ctx:
                await confirm_user_link_token(token="abc", user_id="u1")
        assert ctx.value.status_code == expected_status


class TestDeleteLinkRouteTranslation:
    @pytest.mark.asyncio
    async def test_not_found_maps_to_404(self):
        from backend.api.features.platform_linking.routes import delete_link

        db = _db_mock(
            delete_server_link=AsyncMock(side_effect=NotFoundError("missing"))
        )
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            with pytest.raises(HTTPException) as exc:
                await delete_link(link_id="x", user_id="u1")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_not_owned_maps_to_403(self):
        from backend.api.features.platform_linking.routes import delete_link

        db = _db_mock(
            delete_server_link=AsyncMock(side_effect=NotAuthorizedError("nope"))
        )
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            with pytest.raises(HTTPException) as exc:
                await delete_link(link_id="x", user_id="u1")
        assert exc.value.status_code == 403


class TestDeleteUserLinkRouteTranslation:
    @pytest.mark.asyncio
    async def test_not_found_maps_to_404(self):
        from backend.api.features.platform_linking.routes import delete_user_link_route

        db = _db_mock(delete_user_link=AsyncMock(side_effect=NotFoundError("missing")))
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            with pytest.raises(HTTPException) as exc:
                await delete_user_link_route(link_id="x", user_id="u1")
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_not_owned_maps_to_403(self):
        from backend.api.features.platform_linking.routes import delete_user_link_route

        db = _db_mock(
            delete_user_link=AsyncMock(side_effect=NotAuthorizedError("nope"))
        )
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            with pytest.raises(HTTPException) as exc:
                await delete_user_link_route(link_id="x", user_id="u1")
        assert exc.value.status_code == 403


# ── Adversarial: malformed token path params ──────────────────────────


class TestAdversarialTokenPath:
    # TokenPath enforces `^[A-Za-z0-9_-]+$` + max_length=64.

    @pytest.fixture
    def client(self):
        import fastapi
        from autogpt_libs.auth import get_user_id, requires_user
        from fastapi.testclient import TestClient

        import backend.api.features.platform_linking.routes as routes_mod

        app = fastapi.FastAPI()
        app.dependency_overrides[requires_user] = lambda: None
        app.dependency_overrides[get_user_id] = lambda: "caller-user"
        app.include_router(routes_mod.router, prefix="/api/platform-linking")
        return TestClient(app)

    def test_rejects_token_with_special_chars(self, client):
        response = client.get("/api/platform-linking/tokens/bad%24token/info")
        assert response.status_code == 422

    def test_rejects_token_with_path_traversal(self, client):
        for probe in ("..%2F..", "foo..bar", "foo%2Fbar"):
            response = client.get(f"/api/platform-linking/tokens/{probe}/info")
            assert response.status_code in (
                404,
                422,
            ), f"path-traversal probe {probe!r} returned {response.status_code}"

    def test_rejects_token_too_long(self, client):
        long_token = "a" * 65
        response = client.get(f"/api/platform-linking/tokens/{long_token}/info")
        assert response.status_code == 422

    def test_accepts_token_at_max_length(self, client):
        token = "a" * 64
        db = _db_mock(
            get_link_token_info=AsyncMock(side_effect=NotFoundError("missing"))
        )
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            response = client.get(f"/api/platform-linking/tokens/{token}/info")
        assert response.status_code == 404

    def test_accepts_urlsafe_b64_token_shape(self, client):
        db = _db_mock(
            get_link_token_info=AsyncMock(side_effect=NotFoundError("missing"))
        )
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            response = client.get("/api/platform-linking/tokens/abc-_XYZ123-_abc/info")
        assert response.status_code == 404

    def test_confirm_rejects_malformed_token(self, client):
        response = client.post("/api/platform-linking/tokens/bad%24token/confirm")
        assert response.status_code == 422


class TestAdversarialDeleteLinkId:
    """DELETE link_id has no regex — ensure weird values are handled via
    NotFoundError (no crash, no cross-user leak)."""

    @pytest.fixture
    def client(self):
        import fastapi
        from autogpt_libs.auth import get_user_id, requires_user
        from fastapi.testclient import TestClient

        import backend.api.features.platform_linking.routes as routes_mod

        app = fastapi.FastAPI()
        app.dependency_overrides[requires_user] = lambda: None
        app.dependency_overrides[get_user_id] = lambda: "caller-user"
        app.include_router(routes_mod.router, prefix="/api/platform-linking")
        return TestClient(app)

    def test_weird_link_id_returns_404(self, client):
        db = _db_mock(
            delete_server_link=AsyncMock(side_effect=NotFoundError("missing"))
        )
        with patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=db,
        ):
            for link_id in ("'; DROP TABLE links;--", "../../etc/passwd", ""):
                response = client.delete(f"/api/platform-linking/links/{link_id}")
                assert response.status_code in (404, 405)
