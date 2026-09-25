"""Tests for the Stripe Link confidential-client (authorization-code) handler.

Stripe's contract: https://docs.stripe.com/agentic-commerce/link-agent-wallet/oauth
"""

import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth import HANDLERS_BY_NAME, stripe_link_hosted
from backend.integrations.oauth.stripe_link_hosted import (
    STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED,
    StripeLinkHostedOAuthHandler,
    is_hosted_link_credential,
)

CALLBACK = "https://platform.example/auth/integrations/oauth_callback"


@pytest.fixture
def handler(monkeypatch) -> StripeLinkHostedOAuthHandler:
    monkeypatch.setattr(
        stripe_link_hosted._secrets,
        "stripe_link_publishable_key",
        "pk_test_publishable",
    )
    return StripeLinkHostedOAuthHandler("lwlcid_client", "client-secret", CALLBACK)


@pytest.fixture
def link(monkeypatch):
    """Route every httpx client to a fake login.link.com / api.link.com."""
    requests: list[httpx.Request] = []
    responses: dict[str, httpx.Response] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/userinfo":
            return httpx.Response(200, json={"email": "wallet@example.com"})
        return responses.get(request.url.path) or httpx.Response(
            200,
            json={
                "access_token": "liwltoken_new",
                "refresh_token": f"liwlrefresh_{len(requests)}",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "payment_methods.agentic userinfo:read",
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    return requests, responses


def form(request: httpx.Request) -> dict[str, list[str]]:
    return parse_qs(request.content.decode())


def hosted_credentials(client_id: str = "lwlcid_client") -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="stripe_link",
        access_token=SecretStr("liwltoken_old"),
        refresh_token=SecretStr("liwlrefresh_old"),
        access_token_expires_at=int(time.time()) + 60,
        scopes=["payment_methods.agentic", "userinfo:read"],
        title="Stripe Link",
        metadata={"link_oauth_flow": "authorization_code", "link_client_id": client_id},
    )


def test_registered_only_with_a_configured_client():
    """Unregistered, Stripe Link connects by device code; half a configuration
    must not start a redirect flow Link will reject."""
    assert HANDLERS_BY_NAME.get("stripe_link") is (
        StripeLinkHostedOAuthHandler if STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED else None
    )


def test_login_url_matches_the_documented_authorization_request(handler):
    url = handler.get_login_url([], "state-token", "c" * 43)

    parsed = urlsplit(url)
    params = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://login.link.com/auth"
    )
    assert params == {
        "key": ["pk_test_publishable"],
        "client_id": ["lwlcid_client"],
        "redirect_uri": [CALLBACK],
        "response_type": ["code"],
        "scope": ["payment_methods.agentic userinfo:read"],
        "state": ["state-token"],
        "code_challenge": ["c" * 43],
        "code_challenge_method": ["S256"],
    }
    assert "client-secret" not in url


@pytest.mark.parametrize("state, challenge", [("", "c" * 43), ("state", None)])
def test_login_refuses_to_start_without_state_or_pkce(handler, state, challenge):
    with pytest.raises(ValueError):
        handler.get_login_url([], state, challenge)


@pytest.mark.asyncio
async def test_exchange_authenticates_with_both_keys_and_marks_the_credential(
    handler, link
):
    requests, _ = link

    credentials = await handler.exchange_code_for_tokens(
        "auth-code", ["payment_methods.agentic"], "v" * 128
    )

    token_request = requests[0]
    assert str(token_request.url) == "https://login.link.com/auth/token"
    assert token_request.headers["Authorization"] == "Bearer pk_test_publishable"
    assert form(token_request) == {
        "grant_type": ["authorization_code"],
        "code": ["auth-code"],
        "code_verifier": ["v" * 128],
        "redirect_uri": [CALLBACK],
        "client_id": ["lwlcid_client"],
        "client_secret": ["client-secret"],
    }
    assert credentials.access_token.get_secret_value() == "liwltoken_new"
    assert credentials.refresh_token is not None
    assert credentials.scopes == ["payment_methods.agentic", "userinfo:read"]
    assert credentials.username == "wallet@example.com"
    assert is_hosted_link_credential(credentials)
    assert credentials.metadata["link_client_id"] == "lwlcid_client"


@pytest.mark.asyncio
async def test_exchange_reads_a_comma_delimited_granted_scope(handler, link):
    """link-cli notes the token endpoint echoes `scope` comma-delimited; one
    unsplit string would fail every later `payment_methods.agentic` check."""
    _, responses = link
    responses["/auth/token"] = httpx.Response(
        200,
        json={
            "access_token": "a",
            "refresh_token": "r",
            "expires_in": 3600,
            "scope": "payment_methods.agentic,userinfo:read",
        },
    )

    credentials = await handler.exchange_code_for_tokens("code", [], "v" * 43)

    assert credentials.scopes == ["payment_methods.agentic", "userinfo:read"]


@pytest.mark.asyncio
async def test_exchange_requires_the_pkce_verifier(handler, link):
    requests, _ = link
    with pytest.raises(ValueError):
        await handler.exchange_code_for_tokens("code", [], None)
    assert requests == []


@pytest.mark.asyncio
async def test_refresh_persists_the_rotated_refresh_token(handler, link):
    requests, _ = link
    credentials = hosted_credentials()

    refreshed = await handler.refresh_tokens(credentials)

    assert form(requests[0])["grant_type"] == ["refresh_token"]
    assert form(requests[0])["refresh_token"] == ["liwlrefresh_old"]
    assert refreshed.refresh_token is not None
    assert refreshed.refresh_token.get_secret_value() == "liwlrefresh_1"
    assert refreshed.access_token.get_secret_value() == "liwltoken_new"


@pytest.mark.asyncio
async def test_refresh_reads_a_comma_delimited_granted_scope(handler, link):
    _, responses = link
    responses["/auth/token"] = httpx.Response(
        200,
        json={
            "access_token": "liwltoken_new",
            "refresh_token": "liwlrefresh_new",
            "expires_in": 3600,
            "scope": "payment_methods.agentic,userinfo:read",
        },
    )

    refreshed = await handler.refresh_tokens(hosted_credentials())

    assert refreshed.scopes == ["payment_methods.agentic", "userinfo:read"]


@pytest.mark.asyncio
async def test_a_grant_from_another_client_is_not_refreshed_or_revoked(handler, link):
    requests, _ = link
    credentials = hosted_credentials(client_id="lwlcid_previous")

    with pytest.raises(RuntimeError, match="different OAuth client"):
        await handler.refresh_tokens(credentials)
    # Revocation runs after the local delete, so it reports rather than raises.
    assert await handler.revoke_tokens(credentials) is False
    assert requests == []


@pytest.mark.asyncio
async def test_a_refused_revocation_reports_not_revoked(handler, link):
    _, responses = link
    responses["/auth/revoke"] = httpx.Response(400, json={"error": "invalid_grant"})

    assert await handler.revoke_tokens(hosted_credentials()) is False


@pytest.mark.asyncio
async def test_revoke_ends_the_grant_through_the_refresh_token(handler, link):
    requests, _ = link

    assert await handler.revoke_tokens(hosted_credentials()) is True

    assert str(requests[0].url) == "https://login.link.com/auth/revoke"
    assert form(requests[0])["token"] == ["liwlrefresh_old"]
    assert form(requests[0])["token_type_hint"] == ["refresh_token"]


@pytest.mark.asyncio
async def test_a_failed_call_never_repeats_the_response_body(handler, link):
    """Link can echo the submitted form; the error is persisted and shown."""
    _, responses = link
    responses["/auth/token"] = httpx.Response(
        400, text="client_secret=client-secret liwltoken_canary"
    )

    with pytest.raises(RuntimeError) as failure:
        await handler.exchange_code_for_tokens("code", [], "v" * 43)

    assert "client-secret" not in str(failure.value)
    assert "canary" not in str(failure.value)
    assert "400" in str(failure.value)


def test_device_credentials_are_not_mistaken_for_hosted_ones():
    device = hosted_credentials()
    device.metadata = {}
    assert not is_hosted_link_credential(device)
