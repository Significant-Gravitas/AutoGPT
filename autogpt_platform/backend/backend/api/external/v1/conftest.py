from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import OAuthAccessToken, OAuthApplication

from backend.api.external import rate_limit
from backend.api.external.fastapi_app import external_api
from backend.api.external.v1.routes import UserInfoResponse
from backend.data.auth import api_key
from backend.data.auth.api_key import APIKeyInfoWithHash


@pytest.fixture(scope="module")
def v1_generated_keys():
    return [api_key.keysmith.generate_key() for _ in range(3)]


@pytest.fixture
def v1_key_rows(v1_generated_keys):
    owner = uuid4().hex
    return [
        APIKeyInfoWithHash(
            id=f"key-{i}",
            name="rate-limit-test",
            head=key.head,
            tail=key.tail,
            hash=key.hash,
            salt=key.salt,
            user_id=owner if i < 2 else uuid4().hex,
            scopes=list(APIKeyPermission),
            status=APIKeyStatus.ACTIVE,
            created_at=datetime.now(UTC),
            last_used_at=datetime.now(UTC),
        )
        for i, key in enumerate(v1_generated_keys)
    ]


@pytest.fixture
def v1_tokens(v1_key_rows):
    now = datetime.now(UTC)
    application = OAuthApplication(
        id="app-1",
        name="rate-limit-test",
        clientId="client-1",
        clientSecret="unused",
        clientSecretSalt="unused",
        redirectUris=[],
        grantTypes=["authorization_code", "refresh_token"],
        scopes=list(APIKeyPermission),
        ownerId=v1_key_rows[0].user_id,
        isActive=True,
        createdAt=now,
        updatedAt=now,
    )
    return {
        f"agpt_xt_test-{i}": OAuthAccessToken(
            id=f"token-{i}",
            token=sha256(f"agpt_xt_test-{i}".encode()).hexdigest(),
            applicationId=application.id,
            Application=application,
            userId=row.user_id,
            scopes=list(APIKeyPermission),
            createdAt=now,
            expiresAt=now + timedelta(hours=1),
        )
        for i, row in enumerate(v1_key_rows)
    }


@pytest.fixture
def v1_auth_storage(mocker, v1_key_rows, v1_tokens):
    async def keys_by_head(head):
        return [row for row in v1_key_rows if row.head == head]

    async def token_by_hash(*, where, include):
        return next(
            (row for row in v1_tokens.values() if row.token == where["token"]), None
        )

    key_lookup = mocker.patch.object(
        api_key, "get_active_api_keys_by_head", side_effect=keys_by_head
    )
    token_client = AsyncMock()
    token_client.find_unique.side_effect = token_by_hash
    mocker.patch.object(OAuthAccessToken, "prisma", return_value=token_client)
    return key_lookup, token_client.find_unique


@pytest.fixture
def v1_clock(mocker):
    clock = mocker.patch.object(rate_limit, "datetime")
    clock.now.return_value = datetime(2026, 10, 30, 12, 0, 57, tzinfo=UTC)
    return clock


@pytest.fixture
async def v1_client(mocker, v1_auth_storage, v1_clock):
    async def user_by_id(user_id):
        return UserInfoResponse(
            id=user_id, name="Test user", email="test@example.com", timezone="UTC"
        )

    mocker.patch(
        "backend.api.external.v1.routes.user_db.get_user_by_id",
        side_effect=user_by_id,
    )
    async with AsyncClient(
        transport=ASGITransport(app=external_api), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def v1_counter(mocker):
    counts: dict[str, int] = {}

    async def increment(key, window_seconds):
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    return mocker.patch.object(rate_limit, "_incr_window", side_effect=increment)
