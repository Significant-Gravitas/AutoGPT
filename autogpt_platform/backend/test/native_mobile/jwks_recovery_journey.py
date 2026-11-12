import urllib.request
from test.native_mobile.auth_journey import access_token, auth_service, sign_in_native
from typing import Literal
from unittest.mock import AsyncMock, Mock
from urllib.error import URLError
from urllib.parse import urlencode

import pytest
from autogpt_libs.auth import jwt_utils
from fastapi import WebSocket
from httpx import ASGITransport, AsyncClient
from prisma.models import AuthUser

from backend.api import ws_api
from backend.api.rest_api import app

__all__ = ["auth_service"]


@pytest.mark.parametrize("transport", ["http", "http-body", "websocket"])
async def test_valid_session_recovers_after_jwks_fetch_outage(
    auth_service: tuple[AsyncClient, str],
    monkeypatch: pytest.MonkeyPatch,
    transport: Literal["http", "http-body", "websocket"],
) -> None:
    client, prefix = auth_service
    email = f"{prefix}@example.test"
    bearer = await sign_in_native(client, email)
    token = await access_token(client, bearer)
    user = await AuthUser.prisma().find_unique(where={"email": email})
    assert user is not None
    monkeypatch.setattr(jwt_utils, "_jwks_client", None)
    monkeypatch.setattr(jwt_utils, "_jwks_client_url", None)

    with monkeypatch.context() as outage:
        fetch = Mock(side_effect=URLError("test JWKS outage"))
        outage.setattr(urllib.request.OpenerDirector, "open", fetch)
        status = await _authenticate(transport, token, user.id)
        assert fetch.call_count == 1
        assert status == (1013 if transport == "websocket" else 503)

    recovered = await _authenticate(transport, token, user.id)
    assert recovered == (422 if transport == "http-body" else 200)


async def _authenticate(
    transport: Literal["http", "http-body", "websocket"], token: str, user_id: str
) -> int:
    if transport != "websocket":
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://backend"
        ) as client:
            response = await client.request(
                "POST" if transport == "http-body" else "GET",
                (
                    "/api/internal/graphs"
                    if transport == "http-body"
                    else "/api/internal/auth/user/timezone"
                ),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                content="{not json" if transport == "http-body" else None,
            )
        if response.status_code == 503:
            assert response.headers["Retry-After"] == "5"
        return response.status_code

    send = AsyncMock()
    websocket = WebSocket(
        {"type": "websocket", "query_string": urlencode({"token": token}).encode()},
        receive=AsyncMock(return_value={"type": "websocket.connect"}),
        send=send,
    )
    identity = await ws_api.authenticate_websocket(websocket)
    if identity:
        assert identity == user_id
        send.assert_not_awaited()
        return 200
    assert send.await_args is not None
    return send.await_args.args[0]["code"]
