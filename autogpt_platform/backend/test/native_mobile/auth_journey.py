"""Production Better Auth HTTP, signed JWTs, backend routes, and PostgreSQL."""

import asyncio
import os
import secrets
import shutil
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import urlencode
from uuid import uuid4

import pytest
import pytest_asyncio
from autogpt_libs.auth import config as auth_config
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from prisma.models import AuthSession, AuthUser, User
from starlette.websockets import WebSocket

from backend.api import session_auth, ws_api
from backend.api.features.store import skill_routes
from backend.api.rest_api import app
from backend.data import db


@pytest_asyncio.fixture
async def auth_service(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncClient, str]]:
    if "native_mobile" not in os.environ.get("DATABASE_URL", ""):
        pytest.fail("Use a migrated isolated native_mobile test database")
    node = shutil.which("node")
    assert node is not None
    frontend = Path(__file__).resolve().parents[3] / "frontend"
    prefix = f"native-{uuid4()}"
    env = {
        **os.environ,
        "BETTER_AUTH_SECRET": "native mobile integration test secret only",
        "AUTH_DB_SCHEMA": "platform",
        "AUTH_ALLOW_NEW_ACCOUNTS": "true",
        "AUTH_SIGNUP_ALLOWLIST": "",
        "AUTH_REQUIRE_EMAIL_VERIFICATION": "false",
        "NATIVE_AUTH_PORT": "0",
    }
    process = await asyncio.create_subprocess_exec(
        node,
        "--import",
        "tsx",
        "scripts/native-auth-server.ts",
        cwd=frontend,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    connected = db.is_connected()

    async def wait_for_origin() -> str:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "The production auth harness exited before readiness"
                )
            if line.startswith(b"NATIVE_AUTH_READY "):
                return line.decode().split()[1]

    try:
        origin = await asyncio.wait_for(wait_for_origin(), timeout=40)
        monkeypatch.setenv("JWT_JWKS_URL", f"{origin}/api/auth/jwks")
        monkeypatch.setattr(auth_config, "_settings", auth_config.Settings())
        if not connected:
            await db.connect()
        async with AsyncClient(base_url=origin, timeout=15) as client:
            yield client, prefix
    finally:
        if db.is_connected():
            await AuthUser.prisma().delete_many(where={"email": {"startswith": prefix}})
            await User.prisma().delete_many(where={"email": {"startswith": prefix}})
            if not connected:
                await db.disconnect()
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


async def sign_in_native(client: AsyncClient, email: str) -> str:
    password = secrets.token_urlsafe(24)
    signup = await client.post(
        "/api/auth/sign-up/email",
        json={"name": "Native test", "email": email, "password": password},
    )
    assert signup.status_code == 200
    client.cookies.clear()
    signin = await client.post(
        "/api/auth/sign-in/email", json={"email": email, "password": password}
    )
    assert signin.status_code == 200
    bearer = signin.headers.get("set-auth-token")
    assert bearer is not None
    assert "." in bearer
    client.cookies.clear()
    return bearer


async def access_token(client: AsyncClient, bearer: str) -> str:
    response = await client.get(
        "/api/auth/token", headers={"Authorization": f"Bearer {bearer}"}
    )
    assert response.status_code == 200
    client.cookies.clear()
    return response.json()["token"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_overrides", [False, True])
async def test_native_email_login_renewal_and_signout_revoke_backend_access(
    auth_service: tuple[AsyncClient, str],
    monkeypatch: pytest.MonkeyPatch,
    reset_overrides: bool,
) -> None:
    if reset_overrides:
        monkeypatch.setattr(app, "dependency_overrides", {})
    client, prefix = auth_service
    bearer = await sign_in_native(client, f"{prefix}@example.test")
    token = await access_token(client, bearer)
    renewed = await access_token(client, bearer)
    monkeypatch.setattr(ws_api.settings.config, "enable_auth", True)
    send = AsyncMock()
    websocket = WebSocket(
        {"type": "websocket", "query_string": urlencode({"token": token}).encode()},
        receive=AsyncMock(return_value={"type": "websocket.connect"}),
        send=send,
    )
    assert await ws_api.authenticate_websocket(websocket)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://backend"
    ) as backend:
        for jwt in (token, renewed):
            assert (
                await backend.get(
                    "/api/auth/user/timezone",
                    headers={"Authorization": f"Bearer {jwt}"},
                )
            ).status_code == 200
        response = await client.post(
            "/api/auth/sign-out", headers={"Authorization": f"Bearer {bearer}"}, json={}
        )
        assert response.status_code == 200
        client.cookies.clear()
        assert (
            await client.get(
                "/api/auth/token", headers={"Authorization": f"Bearer {bearer}"}
            )
        ).status_code == 401
        assert await ws_api.authenticate_websocket(websocket) == ""
        assert send.await_args is not None
        assert send.await_args.args[0]["code"] == 4003
        assert (
            await backend.get(
                "/api/auth/user/timezone", headers={"Authorization": f"Bearer {token}"}
            )
        ).status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["expire", "ban", "past_ban", "future_ban", "change_owner"]
)
async def test_backend_rechecks_session_authority(
    auth_service: tuple[AsyncClient, str], change: str
) -> None:
    client, prefix = auth_service
    email = f"{prefix}@example.test"
    bearer = await sign_in_native(client, email)
    token = await access_token(client, bearer)
    user = await AuthUser.prisma().find_unique(where={"email": email})
    assert user is not None
    if change == "expire":
        await AuthSession.prisma().update_many(
            where={"userId": user.id},
            data={"expiresAt": datetime.now(timezone.utc) - timedelta(seconds=1)},
        )
    elif change in {"ban", "past_ban", "future_ban"}:
        await AuthUser.prisma().update(
            where={"id": user.id},
            data={
                "banned": True,
                "banExpires": (
                    None
                    if change == "ban"
                    else datetime.now(timezone.utc)
                    + timedelta(seconds=-60 if change == "past_ban" else 60)
                ),
            },
        )
    else:
        second_email = f"{prefix}-second@example.test"
        await sign_in_native(client, second_email)
        other = await AuthUser.prisma().find_unique(where={"email": second_email})
        assert other is not None
        await AuthSession.prisma().update_many(
            where={"userId": user.id}, data={"userId": other.id}
        )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://backend"
    ) as backend:
        assert (
            await backend.get(
                "/api/auth/user/timezone", headers={"Authorization": f"Bearer {token}"}
            )
        ).status_code == (200 if change == "past_ban" else 401)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unavailable", "timeout"])
async def test_session_lookup_failure_fails_closed_and_recovers(
    auth_service: tuple[AsyncClient, str], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    client, prefix = auth_service
    bearer = await sign_in_native(client, f"{prefix}@example.test")
    token = await access_token(client, bearer)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://backend"
    ) as backend:
        with monkeypatch.context() as outage:

            async def stalled_lookup(*_args: str) -> bool:
                await asyncio.sleep(60)
                return True

            lookup = AsyncMock(
                side_effect=(
                    stalled_lookup if failure == "timeout" else ConnectionError()
                )
            )
            outage.setattr(session_auth, "_SESSION_LOOKUP_TIMEOUT", 0.01)
            outage.setattr(
                session_auth,
                "auth_session_db",
                lambda: AsyncMock(is_auth_session_active=lookup),
            )
            response = await backend.get(
                "/api/auth/user/timezone", headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "5"
        assert (
            await backend.get(
                "/api/auth/user/timezone", headers={"Authorization": f"Bearer {token}"}
            )
        ).status_code == 200


@pytest.mark.asyncio
async def test_native_session_bearer_requires_its_signature(
    auth_service: tuple[AsyncClient, str],
) -> None:
    client, prefix = auth_service
    bearer = await sign_in_native(client, f"{prefix}@example.test")
    unsigned = bearer.split(".", 1)[0]
    assert (
        await client.get(
            "/api/auth/token", headers={"Authorization": f"Bearer {unsigned}"}
        )
    ).status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,lookup_target",
    [
        (
            "/api/store/search?query=private&content_types=CHAT_SESSION",
            "backend.api.features.store.routes.search_engine.unified_hybrid_search",
        ),
        (
            "/api/store/skills/test-skill/files/SKILL.md?version_id=pending",
            "backend.api.features.store.skill_routes.skill_db.find_readable_version",
        ),
    ],
)
async def test_optional_routes_recheck_identity_after_signout(
    auth_service: tuple[AsyncClient, str],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    lookup_target: str,
) -> None:
    monkeypatch.setattr(app, "dependency_overrides", {})
    monkeypatch.setattr(
        skill_routes, "is_feature_enabled", AsyncMock(return_value=True)
    )
    lookup = AsyncMock(side_effect=HTTPException(status_code=404))
    monkeypatch.setattr(lookup_target, lookup)
    client, prefix = auth_service
    email = f"{prefix}@example.test"
    bearer = await sign_in_native(client, email)
    token = await access_token(client, bearer)
    user = await AuthUser.prisma().find_unique(where={"email": email})
    assert user is not None
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://backend"
    ) as backend:
        for expected_user in (user.id, None):
            response = await backend.get(
                path, headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 404
            assert lookup.await_args is not None
            assert lookup.await_args.kwargs["user_id"] == expected_user
            if expected_user is not None:
                response = await client.post(
                    "/api/auth/sign-out",
                    headers={"Authorization": f"Bearer {bearer}"},
                    json={},
                )
                assert response.status_code == 200
