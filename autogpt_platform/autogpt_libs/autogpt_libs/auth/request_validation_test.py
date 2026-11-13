import asyncio
from typing import Annotated
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials
from httpx import ASGITransport, AsyncClient

from autogpt_libs.auth import dependencies, jwt_utils

pytestmark = pytest.mark.asyncio


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    app = FastAPI()
    monkeypatch.setattr(dependencies, "_heal_platform_user", AsyncMock())
    monkeypatch.setattr(
        jwt_utils,
        "parse_jwt_token_async",
        AsyncMock(return_value={"sub": "user", "sid": "session"}),
    )

    @app.get("/combined")
    async def combined(
        optional: Annotated[str | None, Security(dependencies.get_optional_user_id)],
        required: Annotated[str, Security(dependencies.get_user_id)],
    ) -> list[str | None]:
        return [optional, required]

    @app.get("/reversed")
    async def reversed_order(
        required: Annotated[str, Security(dependencies.get_user_id)],
        optional: Annotated[str | None, Security(dependencies.get_optional_user_id)],
    ) -> list[str | None]:
        return [optional, required]

    return app


@pytest.mark.parametrize("route", ["/combined", "/reversed"])
async def test_mixed_dependencies_share_validation_but_next_request_rechecks(
    app: FastAPI, route: str
) -> None:
    validator = AsyncMock()
    app.state.jwt_payload_validator = validator
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.get(route, headers={"Authorization": "Bearer same-token"})
        assert first.status_code == 200
        assert first.json() == ["user", "user"]
        assert validator.await_count == 1
        validator.side_effect = HTTPException(401, "Session revoked")
        revoked = await client.get(
            route, headers={"Authorization": "Bearer same-token"}
        )
        assert revoked.status_code == 401
        assert validator.await_count == 2


@pytest.mark.parametrize("status", [None, 401, 503])
async def test_concurrent_validation_is_shared_including_failures(
    app: FastAPI, status: int | None
) -> None:
    validator = AsyncMock(side_effect=HTTPException(status) if status else None)
    app.state.jwt_payload_validator = validator
    request = Request({"type": "http", "app": app})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="one")
    results = await asyncio.gather(
        jwt_utils.get_jwt_payload(request, credentials),
        jwt_utils.get_jwt_payload(request, credentials),
        return_exceptions=True,
    )
    assert validator.await_count == 1
    if status:
        assert all(
            isinstance(result, HTTPException) and result.status_code == status
            for result in results
        )
    else:
        assert results == [{"sub": "user", "sid": "session"}] * 2


async def test_different_bearers_never_share_validation(app: FastAPI) -> None:
    validator = AsyncMock()
    app.state.jwt_payload_validator = validator
    request = Request({"type": "http", "app": app})
    for token in ("one", "two", "one"):
        await jwt_utils.get_jwt_payload(
            request, HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        )
    assert validator.await_count == 2
