from typing import Annotated
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException, Security
from httpx import ASGITransport, AsyncClient

from autogpt_libs.auth import get_optional_user_id, jwt_utils


@pytest.fixture
def optional_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    app = FastAPI()

    @app.get("/identity")
    async def identity(
        user_id: Annotated[str | None, Security(get_optional_user_id)],
    ) -> str | None:
        return user_id

    monkeypatch.setattr(
        jwt_utils,
        "parse_jwt_token",
        Mock(return_value={"sub": "user", "sid": "session"}),
    )
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("with_validator", [False, True])
async def test_optional_identity_accepts_valid_token(
    optional_app: FastAPI, with_validator: bool
) -> None:
    validator = AsyncMock()
    if with_validator:
        optional_app.state.jwt_payload_validator = validator
    async with AsyncClient(
        transport=ASGITransport(app=optional_app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/identity", headers={"Authorization": "Bearer test"}
        )
    assert response.status_code == 200
    assert response.json() == "user"
    if with_validator:
        validator.assert_awaited_once_with({"sub": "user", "sid": "session"})


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 503])
async def test_optional_identity_respects_session_validation(
    optional_app: FastAPI, status: int
) -> None:
    optional_app.state.jwt_payload_validator = AsyncMock(
        side_effect=HTTPException(status_code=status, headers={"Retry-After": "5"})
    )
    async with AsyncClient(
        transport=ASGITransport(app=optional_app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/identity", headers={"Authorization": "Bearer test"}
        )
    if status == 503:
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"
    else:
        assert response.status_code == 200
        assert response.json() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("with_token", [False, True])
async def test_optional_identity_keeps_anonymous_access(
    optional_app: FastAPI, monkeypatch: pytest.MonkeyPatch, with_token: bool
) -> None:
    validator = AsyncMock()
    optional_app.state.jwt_payload_validator = validator
    monkeypatch.setattr(
        jwt_utils, "parse_jwt_token", Mock(side_effect=ValueError("invalid"))
    )
    async with AsyncClient(
        transport=ASGITransport(app=optional_app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/identity", headers={"Authorization": "Bearer test"} if with_token else {}
        )
    assert response.status_code == 200
    assert response.json() is None
    validator.assert_not_awaited()
