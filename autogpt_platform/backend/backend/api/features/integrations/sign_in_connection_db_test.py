"""A Google sign-up's connection, from the sign-in handoff to the request that
sets the account up, through the real database and Redis.

The handoff comes as the sign-up completes, when the account has only the bare
row the auth hook inserts. A connection is stored in the account's personal
org, which the account's first request makes, so the unit tests (which mock the
store) can't see the order this depends on. Only Google's tokeninfo answer and
the analytics clients are faked.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import fastapi
import httpx
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.models import AuthAccount, AuthUser, Organization, User
from pytest_mock import MockerFixture

from backend.api.features.integrations import sign_in_connection
from backend.api.features.user import routes as user_routes
from backend.api.rest_api import app as rest_app
from backend.data.db import execute_raw_with_schema
from backend.data.model import OAuth2Credentials
from backend.integrations.oauth import google as google_oauth
from backend.integrations.oauth.google import GoogleOAuthHandler
from backend.util.test import SpinTestServer

GMAIL = "https://www.googleapis.com/auth/gmail.readonly"
CALENDAR = "https://www.googleapis.com/auth/calendar"
SIGN_IN = {
    "provider": "google",
    "access_token": "ya29.sign-in-access",
    "refresh_token": "1//sign-in-refresh",
    "expires_at": 2_000_000_000,
    "scope": f"openid {GMAIL} {CALENDAR}",
}

app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
# Mounted where rest_api.py mounts them.
app.include_router(sign_in_connection.sign_in_connection_router, prefix="/api/auth")
app.include_router(user_routes.router, prefix="/api")


class _GoogleTokenInfo:
    """Google's tokeninfo answer for the sign-in's access token."""

    def __init__(self, *args, **kwargs):
        pass

    async def post(self, url, *args, **kwargs):
        return _Answer(
            {
                "azp": "integration-client",
                "aud": "integration-client",
                "email": "pat@gmail.com",
                "scope": SIGN_IN["scope"],
                "expires_in": "3599",
            }
        )


class _Answer:
    ok = True

    def __init__(self, body: dict):
        self._body = body

    def json(self) -> dict:
        return self._body


@pytest.mark.asyncio(loop_scope="session")
async def test_a_google_sign_ups_connection_is_kept_and_the_sign_up_still_counts(
    server: SpinTestServer, mocker: MockerFixture
):
    user_id = str(uuid4())
    email = f"{user_id}@example.com"
    mocker.patch.object(
        sign_in_connection,
        "_get_provider_oauth_handler",
        return_value=GoogleOAuthHandler(
            client_id="integration-client",
            client_secret="secret",
            redirect_uri="https://platform.example/auth/integrations/oauth_callback",
        ),
    )
    mocker.patch.object(google_oauth, "Requests", _GoogleTokenInfo)
    signups = mocker.patch("backend.data.user.track_signup_completed")
    mocker.patch("backend.data.user.set_account_properties")
    mocker.patch("backend.data.user.schedule_posthog_lifecycle_sync")
    mocker.patch(
        "backend.data.tally.populate_understanding_from_tally", new_callable=AsyncMock
    )
    app.dependency_overrides[sign_in_connection.signed_in_user] = lambda: user_id
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": user_id,
        "email": email,
        "role": "user",
    }
    try:
        # What Better Auth and its hook have written when the handoff comes:
        # the identity, its Google account and the bare platform row (the
        # statement in frontend/src/lib/auth/provision-platform-user.ts).
        await AuthUser.prisma().create(
            data={"id": user_id, "name": "Pat", "email": email, "emailVerified": True}
        )
        await AuthAccount.prisma().create(
            data={
                "id": str(uuid4()),
                "accountId": f"google-{user_id}",
                "providerId": "google",
                "userId": user_id,
            }
        )
        await execute_raw_with_schema(
            'INSERT INTO {schema_prefix}"User" (id, email, name, "updatedAt") '
            "VALUES ($1, $2, $3, NOW()) ON CONFLICT (id) DO NOTHING",
            user_id,
            email,
            None,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            handoff = await api.post("/api/auth/sign-in-connection", json=SIGN_IN)
            assert handoff.status_code == 200, handoff.text
            assert handoff.json() == {"result": "held"}
            # The handoff didn't set the account up: its first request does.
            signups.assert_not_called()

            first = await api.post("/api/auth/user")

        assert first.status_code == 200, first.text
        # Still the request that creates the account, which the signup goal
        # and the signup's consent record hang off.
        assert first.headers["X-AutoGPT-User-Created"] == "true"
        signups.assert_called_once_with(user_id=user_id, signup_method="google")
        saved = [
            credentials
            for credentials in await sign_in_connection.creds_manager.store.get_creds_by_provider(
                user_id, "google"
            )
            if isinstance(credentials, OAuth2Credentials)
            and credentials.username == "pat@gmail.com"
        ]
        assert len(saved) == 1
        assert set(saved[0].scopes) == {"openid", GMAIL, CALENDAR}
        assert saved[0].refresh_token is not None
        assert saved[0].refresh_token.get_secret_value() == "1//sign-in-refresh"
    finally:
        app.dependency_overrides.clear()
        # The personal org has no FK to its User, so it would outlive the
        # User's cascade; deleting it takes its members, team and balance.
        await Organization.prisma().delete_many(where={"bootstrapUserId": user_id})
        await User.prisma().delete_many(where={"id": user_id})
        await AuthAccount.prisma().delete_many(where={"userId": user_id})
        await AuthUser.prisma().delete_many(where={"id": user_id})
