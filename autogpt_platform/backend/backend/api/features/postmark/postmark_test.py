"""One-click unsubscribe, answered through the app's own error handlers."""

import base64
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from prisma.actions import UserActions
from prisma.enums import BriefingFrequency, SubscriptionTier
from prisma.errors import DataError
from prisma.models import User as PrismaUser

from backend.api.rest_api import app as rest_app
from backend.data.user import generate_unsubscribe_link

from .postmark import router

app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
app.include_router(router)
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def user_id() -> str:
    # Fresh per test: the user lookup is cached in the shared Redis for minutes.
    return str(uuid.uuid4())


# The User table, beneath the real lookup and preference update.
@pytest.fixture
def lookup(mocker: pytest_mock.MockFixture, user_id: str) -> AsyncMock:
    return mocker.patch.object(
        UserActions, "find_unique", AsyncMock(return_value=_user_row(user_id))
    )


@pytest.fixture
def update(mocker: pytest_mock.MockFixture, user_id: str) -> AsyncMock:
    return mocker.patch.object(
        UserActions, "update", AsyncMock(return_value=_user_row(user_id))
    )


def test_a_valid_token_unsubscribes_its_user(
    lookup: AsyncMock, update: AsyncMock, user_id: str
):
    response = _unsubscribe(_token_for(user_id))

    assert response.status_code == 200
    update.assert_awaited_once()
    assert update.await_args.kwargs["where"] == {"id": user_id}
    assert (
        update.await_args.kwargs["data"]["briefingFrequency"] == BriefingFrequency.OFF
    )


@pytest.mark.parametrize(
    "token",
    [
        "abc",
        "!!!!",
        base64.urlsafe_b64encode(b"no-separator").decode(),
        base64.urlsafe_b64encode(b"\xff\xfe:x").decode(),
        base64.urlsafe_b64encode(b"user-1:not-its-signature").decode(),
        base64.urlsafe_b64encode("user-1:é".encode()).decode(),
    ],
    ids=[
        "bad-padding",
        "not-base64",
        "no-separator",
        "not-utf8",
        "wrong-signature",
        "non-ascii-signature",
    ],
)
def test_a_token_we_did_not_issue_is_a_bad_request(
    token: str, lookup: AsyncMock, update: AsyncMock
):
    response = _unsubscribe(token)

    assert response.status_code == 400
    assert token not in response.text
    update.assert_not_awaited()


def test_a_token_for_a_deleted_user_is_refused_like_a_forged_one(
    lookup: AsyncMock, update: AsyncMock, user_id: str
):
    lookup.return_value = None
    token = _token_for(user_id)

    response = _unsubscribe(token)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid unsubscribe token"
    assert user_id not in response.text and token not in response.text
    update.assert_not_awaited()


def test_a_user_deleted_before_the_update_is_refused_like_a_forged_one(
    lookup: AsyncMock, update: AsyncMock, user_id: str
):
    update.return_value = None
    token = _token_for(user_id)

    response = _unsubscribe(token)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid unsubscribe token"
    assert user_id not in response.text and token not in response.text


def test_a_database_failure_stays_a_server_error(
    lookup: AsyncMock, update: AsyncMock, user_id: str
):
    update.side_effect = DataError({"user_facing_error": {"message": "reset"}})

    response = _unsubscribe(_token_for(user_id))

    assert response.status_code == 500


def _unsubscribe(token: str):
    return client.post("/unsubscribe", params={"token": token})


def _token_for(user_id: str) -> str:
    return parse_qs(urlparse(generate_unsubscribe_link(user_id)).query)["token"][0]


def _user_row(user_id: str) -> PrismaUser:
    now = datetime.now(timezone.utc)
    return PrismaUser(
        id=user_id,
        email="user@example.com",
        emailVerified=True,
        createdAt=now,
        updatedAt=now,
        metadata="{}",
        integrations="",
        maxEmailsPerDay=3,
        briefingFrequency=BriefingFrequency.DAILY,
        alertsEnabled=True,
        notifyOnStoreVerdict=True,
        timezone="UTC",
        subscriptionTier=SubscriptionTier.NO_TIER,
    )
