import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import httpx
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi.routing import APIRoute
from prisma.actions import UserActions
from prisma.models import User as PrismaUser
from pytest_snapshot.plugin import Snapshot

from backend.api.model import RECOGNIZED_TERMS_VERSIONS, UserConsentResponse
from backend.api.rest_api import app as real_app
from backend.data.model import User
from backend.data.notifications import AudienceAction, NotificationResult
from backend.data.user import (
    get_or_create_user,
    get_user_by_email,
    get_user_by_id,
    record_marketing_opt_out_by_email,
    record_signup_consent,
)
from backend.util.exceptions import DatabaseError, NotFoundError

from .routes import router

app = fastapi.FastAPI()
app.include_router(router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, setup_test_user, test_user_id):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload
    from autogpt_libs.auth.models import RequestContext

    # setup_test_user fixture already executed and user is created in database
    # It returns the user_id which we don't need to await

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    # Override get_request_context too — the real one queries Prisma to
    # resolve the user's personal org when no X-Org-Id header is set,
    # which closes/leaks the test event loop across sync TestClient calls.
    async def _fake_request_context() -> RequestContext:
        return RequestContext(
            user_id=test_user_id,
            org_id="test-org",
            team_id=None,
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=False,
            is_team_admin=True,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    app.dependency_overrides[get_request_context] = _fake_request_context
    yield
    app.dependency_overrides.clear()


# Auth endpoints tests
def test_get_or_create_user_route(
    mocker: pytest_mock.MockFixture,
    configured_snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test get or create user endpoint"""
    mock_user = Mock()
    mock_user.created_at = datetime.now(timezone.utc)
    mock_user.model_dump.return_value = {
        "id": test_user_id,
        "email": "test@example.com",
        "name": "Test User",
    }
    mock_result = Mock(user=mock_user, was_created=False)

    mocker.patch(
        "backend.api.features.user.routes.get_or_create_user_with_status",
        return_value=mock_result,
    )

    response = client.post("/auth/user")

    assert response.status_code == 200
    assert response.headers["X-AutoGPT-User-Created"] == "false"
    response_data = response.json()

    configured_snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "auth_user",
    )


def test_get_or_create_user_route_reports_creation(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    mock_user = Mock()
    mock_user.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    mock_user.model_dump.return_value = {
        "id": test_user_id,
        "email": "test@example.com",
    }

    mocker.patch(
        "backend.api.features.user.routes.get_or_create_user_with_status",
        return_value=Mock(user=mock_user, was_created=True),
    )

    response = client.post("/auth/user")

    assert response.status_code == 200
    assert response.headers["X-AutoGPT-User-Created"] == "true"


def test_get_or_create_user_route_documents_creation_header() -> None:
    response_schema = app.openapi()["paths"]["/auth/user"]["post"]["responses"]["200"]

    assert response_schema["headers"]["X-AutoGPT-User-Created"] == {
        "description": "Whether this request created a new user",
        "schema": {"type": "string", "enum": ["true", "false"]},
    }


def test_update_user_email_route(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
) -> None:
    """Test update user email endpoint"""
    mocker.patch(
        "backend.api.features.user.routes.update_user_email",
        return_value=None,
    )

    response = client.post("/auth/user/email", json="newemail@example.com")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["email"] == "newemail@example.com"

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(response_data, indent=2, sort_keys=True),
        "auth_email",
    )


def _consented_user(user_id: str, opted_out_at: datetime | None) -> User:
    accepted_at = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    return User(
        id=user_id,
        email="test@example.com",
        created_at=accepted_at,
        updated_at=accepted_at,
        timezone="Europe/London",
        terms_accepted_at=accepted_at,
        terms_version="2026-10",
        marketing_opt_out_at=opted_out_at,
        marketing_opt_out_source="signup" if opted_out_at else None,
    )


def test_record_user_consent_route(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """Records for the caller's own account, whatever the body claims, and
    answers with the four consent fields only."""
    opted_out_at = datetime(2026, 10, 2, 12, 1, tzinfo=timezone.utc)
    record = mocker.patch(
        "backend.api.features.user.routes.record_signup_consent",
        return_value=_consented_user(test_user_id, opted_out_at),
    )

    response = client.post(
        "/auth/user/consent",
        json={
            "terms_version": "2026-10",
            "marketing_opt_out": True,
            "user_id": "someone-else",
        },
    )

    assert response.status_code == 200
    record.assert_awaited_once_with(test_user_id, "2026-10", True)
    assert response.json() == {
        "terms_accepted_at": "2026-10-02T12:00:00Z",
        "terms_version": "2026-10",
        "marketing_opt_out_at": "2026-10-02T12:01:00Z",
        "marketing_opt_out_source": "signup",
    }


def test_record_user_consent_route_answers_404_for_a_deleted_account(
    mocker: pytest_mock.MockFixture, mock_jwt_user
) -> None:
    """Through the real application, whose handlers turn NotFoundError into a
    404 rather than a 500."""
    mocker.patch(
        "backend.api.features.user.routes.record_signup_consent",
        new=AsyncMock(side_effect=NotFoundError("User not found with ID: x")),
    )
    real_app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    try:
        response = fastapi.testclient.TestClient(real_app).post(
            "/api/auth/user/consent",
            json={"terms_version": "2026-10", "marketing_opt_out": True},
        )
    finally:
        real_app.dependency_overrides.pop(get_jwt_payload, None)

    assert response.status_code == 404


@pytest.mark.parametrize("terms_version", sorted(RECOGNIZED_TERMS_VERSIONS))
def test_record_user_consent_route_accepts_a_recognized_version(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
    terms_version: str,
) -> None:
    record = mocker.patch(
        "backend.api.features.user.routes.record_signup_consent",
        return_value=_consented_user(test_user_id, None),
    )

    response = client.post(
        "/auth/user/consent",
        json={"terms_version": terms_version, "marketing_opt_out": False},
    )

    assert response.status_code == 200
    record.assert_awaited_once_with(test_user_id, terms_version, False)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(None, id="no-body"),
        pytest.param({"marketing_opt_out": False}, id="no-terms-version"),
        pytest.param({"terms_version": "2026-10"}, id="no-marketing-opt-out"),
        pytest.param(
            {"terms_version": "", "marketing_opt_out": False},
            id="empty-terms-version",
        ),
        pytest.param(
            {"terms_version": "   ", "marketing_opt_out": False},
            id="blank-terms-version",
        ),
        pytest.param(
            {"terms_version": "latest", "marketing_opt_out": False},
            id="undated-terms-version",
        ),
        pytest.param(
            {"terms_version": "2026-10; x", "marketing_opt_out": False},
            id="terms-version-with-a-suffix",
        ),
        pytest.param(
            {
                "terms_version": "\u0662\u0660\u0662\u0666-\u0661\u0660",
                "marketing_opt_out": False,
            },
            id="terms-version-in-non-ascii-digits",
        ),
        pytest.param(
            {"terms_version": "v" * 33, "marketing_opt_out": False},
            id="terms-version-over-32-characters",
        ),
        pytest.param(
            {"terms_version": "2026-10-15", "marketing_opt_out": False},
            id="well-formed-but-never-shown",
        ),
        pytest.param(
            {"terms_version": "9999-99", "marketing_opt_out": False},
            id="impossible-date",
        ),
        pytest.param(
            {"terms_version": "2020-01", "marketing_opt_out": False},
            id="older-terms-than-the-page-shows",
        ),
    ],
)
def test_record_user_consent_route_rejects_an_invalid_body(
    mocker: pytest_mock.MockFixture,
    body: dict[str, str | bool] | None,
) -> None:
    record = mocker.patch("backend.api.features.user.routes.record_signup_consent")

    response = client.post("/auth/user/consent", json=body)

    assert response.status_code == 422
    record.assert_not_called()


async def _reset_consent(user_id: str) -> None:
    row = await PrismaUser.prisma().update(
        where={"id": user_id},
        data={
            "termsAcceptedAt": None,
            "termsVersion": None,
            "marketingOptOutAt": None,
            "marketingOptOutSource": None,
        },
    )
    assert row is not None
    get_user_by_id.cache_delete(user_id)
    get_user_by_email.cache_delete(row.email)
    get_or_create_user.cache_clear()


async def _email_of(user_id: str) -> str:
    """The shared test user's stored address. Not necessarily
    test@example.com: the default user may have created the row first."""
    row = await PrismaUser.prisma().find_unique_or_raise(where={"id": user_id})
    return row.email


@pytest.fixture
async def consent_reset(setup_test_user: str) -> AsyncIterator[str]:
    """The test user is shared by the whole session, so the consent this
    module writes onto it is wiped before and after."""
    await _reset_consent(setup_test_user)
    yield setup_test_user
    await _reset_consent(setup_test_user)


def _stored_consent(row: PrismaUser) -> UserConsentResponse:
    return UserConsentResponse(
        terms_accepted_at=row.termsAcceptedAt,
        terms_version=row.termsVersion,
        marketing_opt_out_at=row.marketingOptOutAt,
        marketing_opt_out_source=row.marketingOptOutSource,
    )


async def test_record_user_consent_route_is_idempotent_in_the_database(
    consent_reset: str, mocker: pytest_mock.MockFixture
) -> None:
    """A retried signup write keeps the first stamps, and a later call without
    the opt-out never takes it back. Only the first call, which recorded the
    refusal, unsubscribes the account in MailerLite."""
    queued = mocker.patch(
        "backend.data.user.queue_audience_change",
        new=AsyncMock(return_value=NotificationResult(success=True, message="")),
    )
    consent = {"terms_version": "2026-10", "marketing_opt_out": True}
    where = {"id": consent_reset}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        first = await http.post("/auth/user/consent", json=consent)
        stored = _stored_consent(
            await PrismaUser.prisma().find_unique_or_raise(where=where)
        )
        retried = await http.post("/auth/user/consent", json=consent)
        after_retry = await PrismaUser.prisma().find_unique_or_raise(where=where)
        declined = await http.post(
            "/auth/user/consent", json={**consent, "marketing_opt_out": False}
        )
        after_decline = await PrismaUser.prisma().find_unique_or_raise(where=where)

    assert [first.status_code, retried.status_code, declined.status_code] == [200] * 3
    assert stored.terms_accepted_at is not None
    assert stored.terms_version == "2026-10"
    assert stored.marketing_opt_out_at is not None
    assert stored.marketing_opt_out_source == "signup"
    assert UserConsentResponse.model_validate(first.json()) == stored
    assert _stored_consent(after_retry) == stored
    assert retried.json() == first.json()
    assert _stored_consent(after_decline) == stored
    assert declined.json() == first.json()
    queued.assert_awaited_once()
    assert queued.await_args.args[0].action is AudienceAction.UNSUBSCRIBE
    assert queued.await_args.args[0].user_id == consent_reset


async def test_a_failed_opt_out_write_leaves_neither_half(
    consent_reset: str, mocker: pytest_mock.MockFixture
) -> None:
    """The terms and the opt-out land together or not at all: when the opt-out
    write fails, the terms stamp made before it in the same transaction is
    rolled back, so a retry starts from a clean row."""
    mocker.patch.object(
        UserActions,
        "update_many",
        new=AsyncMock(side_effect=RuntimeError("opt-out write failed")),
    )

    with pytest.raises(DatabaseError):
        await record_signup_consent(consent_reset, "2026-10", True)

    row = await PrismaUser.prisma().find_unique_or_raise(where={"id": consent_reset})
    assert _stored_consent(row) == UserConsentResponse()


async def test_a_mailerlite_unsubscribe_is_recorded_in_the_database(
    consent_reset: str,
) -> None:
    """Matched whatever the address's case, and the first refusal wins: a
    redelivered unsubscribe keeps the first date."""
    where = {"id": consent_reset}
    email = await _email_of(consent_reset)

    first = await record_marketing_opt_out_by_email(email.upper(), "email_unsubscribe")
    stored = await PrismaUser.prisma().find_unique_or_raise(where=where)
    repeated = await record_marketing_opt_out_by_email(email, "email_unsubscribe")
    after = await PrismaUser.prisma().find_unique_or_raise(where=where)
    cached = await get_user_by_id(consent_reset)

    assert first == consent_reset
    assert repeated is None
    assert stored.marketingOptOutAt is not None
    assert stored.marketingOptOutSource == "email_unsubscribe"
    assert _stored_consent(after) == _stored_consent(stored)
    assert cached.marketing_opt_out_at == stored.marketingOptOutAt


async def test_an_unsubscribe_never_matches_another_account_by_wildcard() -> None:
    """`_` is an ILIKE wildcard: an unsubscribe for `john_smith-…` must not
    opt out the account `john.smith-…`."""
    user_id = str(uuid.uuid4())
    await PrismaUser.prisma().create(
        data={"id": user_id, "email": f"john.smith-{user_id}@example.com"}
    )
    try:
        recorded = await record_marketing_opt_out_by_email(
            f"JOHN_SMITH-{user_id}@example.com", "email_unsubscribe"
        )
        row = await PrismaUser.prisma().find_unique_or_raise(where={"id": user_id})
    finally:
        await PrismaUser.prisma().delete_many(where={"id": user_id})

    assert recorded is None
    assert row.marketingOptOutAt is None


async def test_an_unsubscribe_keeps_a_signup_refusal(
    consent_reset: str, mocker: pytest_mock.MockFixture
) -> None:
    mocker.patch(
        "backend.data.user.queue_audience_change",
        new=AsyncMock(return_value=NotificationResult(success=True, message="")),
    )
    await record_signup_consent(consent_reset, "2026-10", True)
    signed_up = await PrismaUser.prisma().find_unique_or_raise(
        where={"id": consent_reset}
    )

    email = await _email_of(consent_reset)

    assert await record_marketing_opt_out_by_email(email, "email_unsubscribe") is None
    after = await PrismaUser.prisma().find_unique_or_raise(where={"id": consent_reset})
    assert after.marketingOptOutSource == "signup"
    assert after.marketingOptOutAt == signed_up.marketingOptOutAt


def test_the_frontend_terms_version_is_recognized() -> None:
    """The signup page sends lib/legal.ts's TERMS_VERSION. A bump there that
    the backend does not recognize would fail every consent write."""
    legal = Path(__file__).parents[5] / "frontend/src/lib/legal.ts"
    shown = re.search(r'TERMS_VERSION = "([^"]+)"', legal.read_text())

    assert shown is not None
    assert shown.group(1) in RECOGNIZED_TERMS_VERSIONS


# Invalid request tests
def test_invalid_json_request() -> None:
    """Test endpoint with invalid JSON"""
    response = client.post(
        "/auth/user/email",
        content="invalid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# The login surface: seven routes authenticate, one deliberately does not.
# Nothing is hoisted onto this router — a router-level dependency would
# silently authenticate the email-link route.
AUTHENTICATED = {
    ("post", "/api/auth/user"),
    ("post", "/api/auth/user/email"),
    ("get", "/api/auth/user/timezone"),
    ("post", "/api/auth/user/timezone"),
    ("post", "/api/auth/user/consent"),
    ("get", "/api/auth/user/preferences"),
    ("post", "/api/auth/user/preferences"),
}
UNAUTHENTICATED = {("post", "/api/auth/user/preferences/from-email")}


@pytest.mark.parametrize("method,path", sorted(AUTHENTICATED))
def test_auth_route_requires_a_user(method: str, path: str):
    """Asserted on the dependency chain, not on `security` in the schema: each
    handler's own Security(get_user_id) or get_jwt_payload puts `security`
    there regardless, so the schema cannot tell requires_user apart."""
    route = next(
        r
        for r in real_app.routes
        if isinstance(r, APIRoute) and r.path == path and method.upper() in r.methods
    )
    assert "requires_user" in {
        d.call.__name__ for d in route.dependant.dependencies if d.call
    }


@pytest.mark.parametrize("method,path", sorted(UNAUTHENTICATED))
def test_email_preference_route_stays_unauthenticated(method: str, path: str):
    """Reached from a link in an email, so it carries no user credential and
    verifies its own signed token instead. If a dependency ever appears here,
    every unsubscribe link in the wild breaks."""
    route = next(
        r
        for r in real_app.routes
        if isinstance(r, APIRoute) and r.path == path and method.upper() in r.methods
    )
    assert route.dependant.dependencies == []
    assert "security" not in real_app.openapi()["paths"][path][method]


def test_auth_surface_has_no_other_operations():
    served = {
        (method.lower(), route.path)
        for route in real_app.routes
        if isinstance(route, APIRoute)
        and route.endpoint.__module__ == "backend.api.features.user.routes"
        for method in route.methods
        if method != "HEAD"
    }
    assert served == AUTHENTICATED | UNAUTHENTICATED


@pytest.mark.parametrize("method,path", sorted(AUTHENTICATED | UNAUTHENTICATED))
def test_auth_operation_keeps_its_tags(method: str, path: str):
    assert real_app.openapi()["paths"][path][method]["tags"] == ["v1", "auth"]
