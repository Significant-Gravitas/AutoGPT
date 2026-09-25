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

import logging
import time
from typing import ClassVar, Optional
from urllib.parse import urlencode

import httpx
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.base import BaseOAuthHandler, parse_granted_scopes
from backend.integrations.oauth.stripe_link import (
    LINK_AUTH_BASE_URL,
    LINK_HTTP_TIMEOUT,
    fetch_link_username,
)
from backend.integrations.providers import ProviderName
from backend.util.settings import Secrets

logger = logging.getLogger(__name__)

_secrets = Secrets()
STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED = bool(
    _secrets.stripe_link_client_id
    and _secrets.stripe_link_client_secret
    and _secrets.stripe_link_publishable_key
)

HOSTED_FLOW = "authorization_code"


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
            access_token_expires_at=int(time.time()) + int(data["expires_in"]),
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
        except (RuntimeError, httpx.HTTPError) as e:
            logger.warning(f"Stripe Link grant was not revoked: {type(e).__name__}")
            return False
        return True

    def _require_own_client(self, credentials: OAuth2Credentials) -> None:
        # Tokens are bound to the client that issued them; after a client
        # rotation the old grant cannot be refreshed and must be reconnected.
        if not self._owns(credentials):
            raise RuntimeError(
                "These Stripe Link credentials were issued to a different "
                "OAuth client; reconnect Stripe Link"
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
            # message is persisted and shown to users.
            logger.warning(
                "Stripe Link %s request failed with HTTP %s",
                endpoint,
                response.status_code,
            )
            raise RuntimeError(
                f"Stripe Link authorization failed (HTTP {response.status_code})"
            )
        return {} if endpoint == "revoke" else response.json()
