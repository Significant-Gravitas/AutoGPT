"""v2 credential create keeps an API key's expiry and refuses a past one (#16234)."""

import time
from datetime import datetime, timezone
from typing import Annotated
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from fastapi import Body, HTTPException
from prisma.enums import APIKeyPermission
from pydantic import TypeAdapter, ValidationError

from backend.data.model import APIKeyCredentials

from ..models import CredentialCreateRequest
from ..tenancy import TenantContext
from .credentials import create_credential

_AUTH = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-1",
)

_REQUEST = TypeAdapter(Annotated[CredentialCreateRequest, Body(discriminator="type")])


@pytest.fixture
def create(mocker: pytest_mock.MockFixture) -> AsyncMock:
    return mocker.patch(
        "backend.api.external.v2.integrations.credentials.creds_manager.create",
        new_callable=AsyncMock,
    )


def _body(expires_at: str | None) -> dict:
    body = {"type": "api_key", "provider": "openai", "api_key": "sk-1"}
    if expires_at is not None:
        body["expires_at"] = expires_at
    return body


async def test_the_expiry_is_stored_and_returned(create: AsyncMock) -> None:
    expires = datetime.fromtimestamp(int(time.time()) + 86400, tz=timezone.utc)

    info = await create_credential(
        request=_REQUEST.validate_python(_body(expires.isoformat())), auth=_AUTH
    )

    stored = create.await_args.args[1]
    assert isinstance(stored, APIKeyCredentials)
    assert stored.expires_at == int(expires.timestamp())
    assert info.expires_at == expires


async def test_no_expiry_stays_none(create: AsyncMock) -> None:
    info = await create_credential(
        request=_REQUEST.validate_python(_body(None)), auth=_AUTH
    )

    assert create.await_args.args[1].expires_at is None
    assert info.expires_at is None


async def test_a_past_expiry_is_refused(create: AsyncMock) -> None:
    with pytest.raises(HTTPException) as exc:
        await create_credential(
            request=_REQUEST.validate_python(_body("2020-01-01T00:00:00Z")),
            auth=_AUTH,
        )

    assert exc.value.status_code == 422
    assert exc.value.detail == "Expiry must be in the future"
    create.assert_not_awaited()


def test_an_expiry_without_a_time_zone_is_refused() -> None:
    with pytest.raises(ValidationError):
        _REQUEST.validate_python(_body("2999-01-01T00:00:00"))
