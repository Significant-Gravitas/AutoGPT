"""
Stripe Link — OAuth 2.0 authorization-code handler for hosted deployments.

The device-code grant in ``stripe_link.py`` uses Link's public CLI client, which
suits self-hosted installs. Stripe documents a different identity for a hosted
agent service: a registered *confidential* client (client ID and secret, exact
redirect URIs) using the authorization-code flow with S256 PKCE, refresh-token
rotation and revocation.
https://docs.stripe.com/agentic-commerce/link-agent-wallet/oauth

Every call to ``login.link.com`` authenticates twice: the publishable key of
the Stripe account that registered the client goes in the ``Authorization``
header, and the client secret goes in the form body.

When the three ``STRIPE_LINK_*`` settings are present, the handler is
registered and Stripe Link connects through it; otherwise it is left out of the
registry and Stripe Link connects by device code. Credentials it issues are
marked in their metadata, so refresh and revocation keep reaching the client
that issued them even while older device-code credentials remain in use.
"""

import json
import logging
import re
import time
from typing import ClassVar, Optional
from urllib.parse import urlencode

import httpx
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.base import BaseOAuthHandler, parse_granted_scopes
from backend.integrations.oauth.refresh_failure import OAuthTokenRequestError
from backend.integrations.oauth.stripe_link import (
    HOSTED_FLOW,
    LINK_AUTH_BASE_URL,
    LINK_HTTP_TIMEOUT,
    StripeLinkDeviceAuthHandler,
    fetch_link_username,
)
from backend.integrations.providers import ProviderName
from backend.util.settings import Config, Secrets

logger = logging.getLogger(__name__)

_config = Config()
_secrets = Secrets()
STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED = bool(
    _secrets.stripe_link_client_id
    and _secrets.stripe_link_client_secret
    and _secrets.stripe_link_publishable_key
)

# What an OAuth error code looks like (RFC 6749 section 5.2).
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")

# The financial-insights grant: the source actions the device-code flow asks
# for, as the JSON array the authorization endpoint takes in one
# `authorization_details` parameter (the device endpoint reads the same detail
# as bracketed form fields). Compact, so the URL carries no encoded spaces.
# https://docs.stripe.com/financial-connections/agents/financial-insights
AUTHORIZATION_DETAILS = json.dumps(
    [{"type": "source", "actions": StripeLinkDeviceAuthHandler.SOURCE_ACTIONS}],
    separators=(",", ":"),
)


def requests_financial_insights() -> bool:
    """Whether the hosted login also asks the customer to share bank and card
    accounts with the financial-insights blocks.

    Opt-in, because Link refuses the whole authorization, payments included,
    when it cannot grant something asked for: insights need the Stripe account
    registered for Financial Connections, and Link offers them to US consumers
    only, while agent payments also serve Canada.
    """
    return _config.stripe_link_financial_insights


def is_hosted_link_credential(credentials: OAuth2Credentials) -> bool:
    """Whether these credentials came from the confidential client, which is
    the only one that can refresh or revoke them."""
    return credentials.metadata.get("link_oauth_flow") == HOSTED_FLOW


class StripeLinkHostedOAuthHandler(BaseOAuthHandler):
    PROVIDER_NAME: ClassVar[ProviderName | str] = ProviderName.STRIPE_LINK
    DEFAULT_SCOPES: ClassVar[list[str]] = ["payment_methods.agentic", "userinfo:read"]
    # Every refresh returns a new refresh token and invalidates the old one.
    ROTATES_REFRESH_TOKEN: ClassVar[bool] = True

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.publishable_key = _secrets.stripe_link_publishable_key

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        # Link rejects the request without both; failing here names the cause.
        if not state or not code_challenge:
            raise ValueError("Stripe Link requires a state token and a PKCE challenge")
        params = {
            "key": self.publishable_key,
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.handle_default_scopes(scopes)),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if requests_financial_insights():
            params["authorization_details"] = AUTHORIZATION_DETAILS
        return f"{LINK_AUTH_BASE_URL}/auth?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        if not code_verifier:
            raise ValueError("Stripe Link requires the PKCE code verifier")
        data = await self._post(
            "token",
            {
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": code_verifier,
                "redirect_uri": self.redirect_uri,
            },
        )
        access_token = data["access_token"]
        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title="Stripe Link",
            access_token=SecretStr(access_token),
            refresh_token=SecretStr(data["refresh_token"]),
            access_token_expires_at=(
                int(time.time()) + int(data["expires_in"])
                if data.get("expires_in") is not None
                else None
            ),
            # What the customer granted, which can be a subset of the request;
            # no fallback to the requested scopes (see the device handler).
            scopes=parse_granted_scopes(data.get("scope"), []),
            username=await fetch_link_username(access_token),
            metadata={"link_oauth_flow": HOSTED_FLOW, "link_client_id": self.client_id},
        )

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise RuntimeError("No refresh token available")
        self._require_own_client(credentials)
        data = await self._post(
            "token",
            {
                "grant_type": "refresh_token",
                "refresh_token": credentials.refresh_token.get_secret_value(),
            },
        )
        credentials.access_token = SecretStr(data["access_token"])
        if data.get("refresh_token"):
            credentials.refresh_token = SecretStr(data["refresh_token"])
        if data.get("expires_in") is not None:
            credentials.access_token_expires_at = int(time.time()) + int(
                data["expires_in"]
            )
        if data.get("scope"):
            credentials.scopes = parse_granted_scopes(data["scope"], credentials.scopes)
        return credentials

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        # Called after the local delete, so a failure is reported as "not
        # revoked" rather than raised: raising would turn a completed
        # disconnect into an error. The customer can still end the grant from
        # their Link account.
        if not credentials.refresh_token:
            return False
        if not self._owns(credentials):
            logger.warning(
                "Stripe Link grant was issued to another client; not revoked"
            )
            return False
        # Revoking the refresh token ends the grant, and with it every access
        # token issued from it.
        try:
            await self._post(
                "revoke",
                {
                    "token": credentials.refresh_token.get_secret_value(),
                    "token_type_hint": "refresh_token",
                },
            )
        except (OAuthTokenRequestError, RuntimeError, httpx.HTTPError) as e:
            logger.warning(f"Stripe Link grant was not revoked: {type(e).__name__}")
            return False
        return True

    def _require_own_client(self, credentials: OAuth2Credentials) -> None:
        if not is_hosted_link_credential(credentials):
            # A reconnect through the device client replaced the grant after
            # this handler was picked. Not a dead grant: the next refresh
            # picks that client.
            raise RuntimeError("These Stripe Link credentials are not this client's")
        if not self._owns(credentials):
            # Issued to a confidential client since rotated or removed, which
            # nothing here can refresh: a grant "issued to another client" is
            # RFC 6749's invalid_grant, so the user is asked to reconnect.
            raise OAuthTokenRequestError(
                "Stripe Link", status_code=None, error_code="invalid_grant"
            )

    def _owns(self, credentials: OAuth2Credentials) -> bool:
        return credentials.metadata.get("link_client_id") == self.client_id

    async def _post(self, endpoint: str, form: dict[str, str]) -> dict:
        async with httpx.AsyncClient(timeout=LINK_HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{LINK_AUTH_BASE_URL}/auth/{endpoint}",
                headers={"Authorization": f"Bearer {self.publishable_key}"},
                data={
                    **form,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
        if response.status_code != 200:
            # The body can echo the submitted form, secret included, and the
            # message is persisted and shown to users: only the status and
            # OAuth's error code leave here, which is what lets the refresh
            # path tell a dead grant (invalid_grant) from a passing failure.
            logger.warning(
                "Stripe Link %s request failed with HTTP %s",
                endpoint,
                response.status_code,
            )
            raise OAuthTokenRequestError(
                "Stripe Link",
                status_code=response.status_code,
                error_code=_oauth_error_code(response),
            )
        return {} if endpoint == "revoke" else response.json()


def _oauth_error_code(response: httpx.Response) -> str | None:
    """RFC 6749's ``error`` field, if the body is the JSON error it describes.

    The code is persisted and shown in error messages, so anything not shaped
    like one (the same rule ``refresh_failure`` applies) is dropped."""
    try:
        body = response.json()
    except ValueError:
        return None
    code = body.get("error") if isinstance(body, dict) else None
    return code if isinstance(code, str) and _ERROR_CODE.fullmatch(code) else None


def unconfigured_handler() -> StripeLinkHostedOAuthHandler:
    """Stands in for the confidential client once it is no longer configured.
    A grant it issued keeps working while its access token lasts; its refresh
    then fails as a dead grant, so the user is asked to reconnect."""
    return StripeLinkHostedOAuthHandler("", "", redirect_uri="")


def revocation_handler() -> StripeLinkHostedOAuthHandler:
    """The configured client, built to end a grant. Revoking sends no redirect
    URI, so a deployment without a frontend URL can still revoke."""
    return StripeLinkHostedOAuthHandler(
        _secrets.stripe_link_client_id,
        _secrets.stripe_link_client_secret,
        redirect_uri="",
    )
