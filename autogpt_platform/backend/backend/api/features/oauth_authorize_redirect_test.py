from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from prisma.enums import APIKeyPermission

from backend.api.features import oauth
from backend.data.auth.oauth import OAuthApplicationInfo

REGISTERED = "https://client.example/callback"
UNREGISTERED = "https://evil.example/collect"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "app_state, response_type",
    [("unknown", "code"), ("inactive", "code"), ("active", "token")],
)
async def test_an_error_found_before_the_redirect_uri_checks_out_is_not_redirected(
    app_state, response_type
):
    with pytest.raises(HTTPException) as error:
        await _authorize(app_state, UNREGISTERED, response_type=response_type)

    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_an_unexpected_failure_before_the_check_is_not_redirected():
    failing_lookup = AsyncMock(side_effect=RuntimeError("database unavailable"))
    with patch.object(oauth, "get_oauth_application", failing_lookup), pytest.raises(
        HTTPException
    ) as error:
        await oauth.authorize(request=_request(UNREGISTERED), user_id="user")

    assert error.value.status_code == 500


@pytest.mark.asyncio
async def test_an_error_found_once_the_redirect_uri_checks_out_goes_back_to_it():
    response = await _authorize("active", REGISTERED, response_type="token")

    sent_to = urlparse(response.redirect_url)
    assert f"{sent_to.scheme}://{sent_to.netloc}{sent_to.path}" == REGISTERED
    assert parse_qs(sent_to.query)["error"] == ["unsupported_response_type"]


@pytest.mark.asyncio
async def test_an_unexpected_failure_once_the_redirect_uri_checks_out_goes_back_to_it():
    failing_issue = AsyncMock(side_effect=RuntimeError("database unavailable"))
    with patch.object(oauth, "create_authorization_code", failing_issue):
        response = await _authorize("active", REGISTERED, response_type="code")

    sent_to = urlparse(response.redirect_url)
    assert f"{sent_to.scheme}://{sent_to.netloc}{sent_to.path}" == REGISTERED
    assert parse_qs(sent_to.query)["error"] == ["server_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("challenge", "method"),
    [
        ("too-short", "S256"),
        ("a" * 44, "S256"),
        ("short-plain", "plain"),
        ("x" * 129, "plain"),
        ("not allowed!" * 5, "plain"),
    ],
)
async def test_a_malformed_challenge_goes_back_to_the_client(
    challenge: str, method: str
):
    """Once the client and redirect URI check out, a bad PKCE challenge is the
    client's error to receive (RFC 6749 4.1.2.1), not a 422 it never sees."""
    app = _app(is_active=True)
    with patch.object(oauth, "get_oauth_application", AsyncMock(return_value=app)):
        response = await oauth.authorize(
            request=_request(
                REGISTERED, code_challenge=challenge, code_challenge_method=method
            ),
            user_id="user",
        )

    sent_to = urlparse(response.redirect_url)
    assert f"{sent_to.scheme}://{sent_to.netloc}{sent_to.path}" == REGISTERED
    assert parse_qs(sent_to.query)["error"] == ["invalid_request"]
    assert parse_qs(sent_to.query)["state"] == ["state-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("in_rollout", [False, True])
async def test_only_the_local_pc_rollout_can_link_a_computer(in_rollout: bool):
    """Every Local PC route and connection checks the rollout flag, but
    signing the companion in didn't: a user outside it got a token that
    every connection then refused with 4403. The scope stops at sign-in now,
    with the reason for the companion to show."""
    app = _app(is_active=True).model_copy(
        update={"scopes": [APIKeyPermission.CONNECT_LOCAL_EXECUTOR]}
    )
    issue = AsyncMock(return_value=SimpleNamespace(code="code-1"))
    gate = AsyncMock(return_value=in_rollout)
    with (
        patch.object(oauth, "get_oauth_application", AsyncMock(return_value=app)),
        patch.object(oauth, "is_local_executor_enabled", gate),
        patch.object(oauth, "refuse_closed_account", AsyncMock()),
        patch.object(oauth, "create_authorization_code", issue),
    ):
        response = await oauth.authorize(
            request=_request(
                REGISTERED, scopes=[APIKeyPermission.CONNECT_LOCAL_EXECUTOR.value]
            ),
            user_id="user",
        )

    query = parse_qs(urlparse(response.redirect_url).query)
    gate.assert_awaited_once_with("user")
    if in_rollout:
        assert query["code"] == ["code-1"]
    else:
        assert query["error"] == ["access_denied"]
        assert query["error_description"] == [
            "Local PC isn't available for this account yet."
        ]
        assert query["state"] == ["state-1"]
        issue.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_scopes_never_read_the_local_pc_flag():
    gate = AsyncMock(return_value=False)
    issue = AsyncMock(return_value=SimpleNamespace(code="code-1"))
    with (
        patch.object(
            oauth,
            "get_oauth_application",
            AsyncMock(return_value=_app(is_active=True)),
        ),
        patch.object(oauth, "is_local_executor_enabled", gate),
        patch.object(oauth, "refuse_closed_account", AsyncMock()),
        patch.object(oauth, "create_authorization_code", issue),
    ):
        response = await oauth.authorize(request=_request(REGISTERED), user_id="user")

    assert parse_qs(urlparse(response.redirect_url).query)["code"] == ["code-1"]
    gate.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_closed_account_gets_no_code():
    """Asked of the database: for a few seconds after the close, another
    worker's cache can still let an old token through."""
    closed = HTTPException(status_code=401, detail="This account is closed.")
    issue = AsyncMock()
    with patch.object(oauth, "create_authorization_code", issue), pytest.raises(
        HTTPException
    ) as error:
        await _authorize(
            "active",
            REGISTERED,
            response_type="code",
            closed_check=AsyncMock(side_effect=closed),
        )

    assert error.value.status_code == 401
    issue.assert_not_awaited()


async def _authorize(
    app_state: str,
    redirect_uri: str,
    *,
    response_type: str,
    closed_check: AsyncMock | None = None,
):
    app = None if app_state == "unknown" else _app(is_active=app_state == "active")
    with patch.object(
        oauth, "get_oauth_application", AsyncMock(return_value=app)
    ), patch.object(oauth, "refuse_closed_account", closed_check or AsyncMock()):
        return await oauth.authorize(
            request=_request(redirect_uri, response_type=response_type),
            user_id="user",
        )


def _request(
    redirect_uri: str,
    *,
    response_type: str = "code",
    # RFC 7636 appendix B: a valid S256 challenge.
    code_challenge: str = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
    code_challenge_method: str = "S256",
    scopes: list[str] | None = None,
):
    return oauth.AuthorizeRequest(
        client_id="client-123",
        redirect_uri=redirect_uri,
        scopes=scopes or [APIKeyPermission.READ_GRAPH.value],
        state="state-1",
        response_type=response_type,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
    )


def _app(*, is_active: bool) -> OAuthApplicationInfo:
    now = datetime.now(UTC)
    return OAuthApplicationInfo(
        id="app-123",
        name="Test app",
        client_id="client-123",
        redirect_uris=[REGISTERED],
        grant_types=["authorization_code"],
        scopes=[APIKeyPermission.READ_GRAPH],
        owner_id="owner",
        is_active=is_active,
        created_at=now,
        updated_at=now,
    )
