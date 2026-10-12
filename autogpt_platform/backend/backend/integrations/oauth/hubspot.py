import logging
import time
from typing import Any, ClassVar, Optional
from urllib.parse import urlencode

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.util.request import Requests, Response

from .base import BaseOAuthHandler, parse_granted_scopes

logger = logging.getLogger(__name__)


class HubSpotOAuthError(Exception):
    """HubSpot's token endpoint refused a request."""


class HubSpotOAuthHandler(BaseOAuthHandler):
    """
    Based on the documentation at:
    - [Manage OAuth tokens (2026-09)](https://developers.hubspot.com/docs/api-reference/latest/authentication/manage-oauth-tokens)
    - [Revoke a token](https://developers.hubspot.com/docs/api-reference/latest/authentication/oauth-tokens/revoke-token)
    - [Scopes](https://developers.hubspot.com/docs/apps/developer-platform/build-apps/authentication/scopes)

    Notes:
    - Access tokens expire after 30 minutes; the refresh token is long-lived.
    - Every scope ticked in the app's Auth settings is required, and HubSpot shows
      an error page if the authorize URL leaves one out. The login URL therefore
      always asks for the full default set, plus anything a block adds.
    - The date-versioned endpoints take every token and secret in the form body,
      never the URL. The v1 endpoints, which put tokens in the path, are retired
      on 2027-02-16. Revoking deletes the refresh token; access tokens already
      issued stay valid until they expire.
    """

    PROVIDER_NAME = ProviderName.HUBSPOT
    # What the HubSpot blocks call: the contacts and companies CRM objects, and
    # email engagements, which HubSpot accepts under the contacts scopes.
    # `oauth` is added to every HubSpot app and so must be requested.
    DEFAULT_SCOPES: ClassVar[list[str]] = [
        "oauth",
        "crm.objects.contacts.read",
        "crm.objects.contacts.write",
        "crm.objects.companies.read",
        "crm.objects.companies.write",
    ]

    AUTHORIZE_URL = "https://app.hubspot.com/oauth/authorize"
    TOKEN_URL = "https://api.hubspot.com/oauth/2026-09/token"
    INTROSPECT_URL = "https://api.hubspot.com/oauth/2026-09/token/introspect"
    REVOKE_URL = "https://api.hubapi.com/oauth/2026-09/token/revoke"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: Optional[str]
    ) -> str:
        requested = list(self.DEFAULT_SCOPES)
        requested += [s for s in scopes if s not in requested]
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(requested),
            "state": state,
        }
        return f"{self.AUTHORIZE_URL}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: Optional[str]
    ) -> OAuth2Credentials:
        tokens = await self._request_tokens(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
            }
        )
        token_info = await self._introspect(tokens["access_token"])
        hub_id = tokens.get("hub_id") or token_info.get("hub_id")
        return OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=token_info.get("hub_domain"),
            username=_portal_username(token_info.get("user"), hub_id),
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
            access_token_expires_at=self._expires_at(tokens),
            refresh_token_expires_at=None,
            scopes=self._granted_scopes(tokens, fallback=scopes),
            metadata={"hub_id": hub_id},
        )

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise ValueError("HubSpot credentials have no refresh token")

        tokens = await self._request_tokens(
            {
                "grant_type": "refresh_token",
                "refresh_token": credentials.refresh_token.get_secret_value(),
            }
        )
        return OAuth2Credentials(
            id=credentials.id,
            provider=self.PROVIDER_NAME,
            title=credentials.title,
            username=credentials.username,
            access_token=tokens["access_token"],
            # HubSpot normally returns the same refresh token; keep the old one
            # if a response ever leaves it out.
            refresh_token=tokens.get("refresh_token") or credentials.refresh_token,
            access_token_expires_at=self._expires_at(tokens),
            refresh_token_expires_at=credentials.refresh_token_expires_at,
            scopes=self._granted_scopes(tokens, fallback=credentials.scopes),
            metadata=credentials.metadata,
        )

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not credentials.refresh_token:
            return False

        # The credential is already deleted on our side by the time this runs,
        # so no failure here may turn the deletion into an error.
        try:
            response = await _requests().post(
                self.REVOKE_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "token_type_hint": "refresh_token",
                    "token": credentials.refresh_token.get_secret_value(),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                allow_redirects=False,
            )
        except Exception as e:
            logger.warning("Could not reach HubSpot to revoke a refresh token: %s", e)
            return False
        if not response.ok:
            logger.warning(
                "HubSpot refused to revoke a refresh token (HTTP %s)", response.status
            )
            return False
        return True

    async def _request_tokens(self, params: dict[str, str]) -> dict[str, Any]:
        response = await _requests().post(
            self.TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                **params,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            # The body carries the client secret: never re-send it elsewhere.
            allow_redirects=False,
        )
        if not response.ok:
            raise HubSpotOAuthError(
                f"HubSpot token request failed (HTTP {response.status}): "
                f"{_error_message(response)}"
            )
        tokens = response.json()
        if not tokens.get("access_token"):
            raise HubSpotOAuthError("HubSpot token response has no access token")
        return tokens

    async def _introspect(self, access_token: str) -> dict[str, Any]:
        """The HubSpot user and account a token belongs to, or {} if unknown."""
        try:
            response = await _requests().post(
                self.INTROSPECT_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "token_type_hint": "access_token",
                    "token": access_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                allow_redirects=False,
            )
            if response.ok:
                return response.json()
            logger.warning(
                "HubSpot token introspection failed (HTTP %s)", response.status
            )
        except Exception as e:
            # Only labels the credential; the sign-in itself has succeeded.
            logger.warning("HubSpot token introspection failed: %s", e)
        return {}

    @staticmethod
    def _expires_at(tokens: dict[str, Any]) -> Optional[int]:
        expires_in = tokens.get("expires_in")
        return int(time.time()) + int(expires_in) if expires_in else None

    @staticmethod
    def _granted_scopes(tokens: dict[str, Any], fallback: list[str]) -> list[str]:
        # HubSpot returns the granted scopes as a list under `scopes`.
        granted = " ".join(tokens.get("scopes") or [])
        return parse_granted_scopes(granted, fallback=fallback)


def _portal_username(user: Optional[str], hub_id: Any) -> Optional[str]:
    """The HubSpot user *in one portal*.

    The OAuth callback merges a new sign-in into an existing credential with the
    same username. One HubSpot login can belong to several portals, so the bare
    email would let connecting a second portal overwrite the first one's tokens.
    """
    if not user:
        return None
    return f"{user} (portal {hub_id})" if hub_id else user


def _requests() -> Requests:
    # Without a cap, a throttled (429) or failing (5xx) HubSpot is retried
    # forever and the sign-in or block run hangs. After the last attempt the
    # response is returned and reported like any other failure.
    return Requests(raise_for_status=False, retry_max_attempts=3)


def _error_message(response: Response) -> str:
    try:
        body = response.json()
    except Exception:
        return response.text()[:200]
    return str(
        body.get("error_description") or body.get("message") or body.get("error")
    )
