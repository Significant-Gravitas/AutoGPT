"""One Microsoft connection for mail, calendar, files, Teams, Copilot and Work IQ.

A deployment with an Entra app secret signs people in through the browser
(authorization code with PKCE). Without one, such as a fresh local install,
the same connection comes from a sign-in code against AutoGPT's public client.
Both paths share the token code below and produce the same credential, so
everything downstream only ever sees provider ``microsoft``.

A sign-in may ask consent for Graph and Work IQ at once, but a token request
names one resource. The stored access token is always Graph; Work IQ tokens
are minted from the same refresh token when needed.
"""

from urllib.parse import urlencode

import httpx

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.integrations.oauth.device_base import (
    BaseDeviceAuthHandler,
    DeviceAuthInitiation,
    DeviceAuthPollResult,
)
from backend.integrations.oauth.microsoft_recovery import (
    MicrosoftAuthError,
    microsoft_recovery_from_response,
)
from backend.integrations.oauth.microsoft_services import (
    DEFAULT_MICROSOFT_SCOPES,
    consent_scopes,
    token_scopes,
)
from backend.integrations.oauth.microsoft_tokens import (
    LOGIN_BASE,
    TIMEOUT_SECONDS,
    client_fields,
    credentials_from_tokens,
    error_code,
    microsoft_app,
    refresh_microsoft_credentials,
    token_request,
)
from backend.integrations.providers import ProviderName


class MicrosoftOAuthHandler(BaseOAuthHandler):
    """Browser sign-in, for deployments whose Entra app has a secret."""

    PROVIDER_NAME = ProviderName.MICROSOFT
    DEFAULT_SCOPES = DEFAULT_MICROSOFT_SCOPES

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_login_url(
        self, scopes: list[str], state: str, code_challenge: str | None
    ) -> str:
        if not code_challenge:
            raise ValueError("Microsoft sign-in requires PKCE.")
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "response_mode": "query",
            "scope": " ".join(consent_scopes(scopes)),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            # Personal and work accounts can share an address, so the person
            # always picks which one this connection is.
            "prompt": "select_account",
        }
        return f"{LOGIN_BASE}/common/oauth2/v2.0/authorize?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, code: str, scopes: list[str], code_verifier: str | None
    ) -> OAuth2Credentials:
        if not code_verifier:
            raise ValueError("Microsoft sign-in requires a PKCE verifier.")
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(token_scopes(consent_scopes(scopes))),
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        payload = await token_request("common", data)
        return await credentials_from_tokens(
            payload, client_id=self.client_id, auth_flow="authorization_code"
        )

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        return await refresh_microsoft_credentials(credentials)

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        return False


class MicrosoftDeviceAuthHandler(BaseDeviceAuthHandler):
    """Sign-in codes, for installs without an Entra app secret."""

    PROVIDER_NAME = ProviderName.MICROSOFT
    DEFAULT_SCOPES = DEFAULT_MICROSOFT_SCOPES

    async def initiate_device_auth(self, scopes: list[str]) -> DeviceAuthInitiation:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{LOGIN_BASE}/common/oauth2/v2.0/devicecode",
                data={
                    "client_id": microsoft_app().client_id,
                    "scope": " ".join(consent_scopes(scopes)),
                },
            )
        if not response.is_success:
            raise MicrosoftAuthError(microsoft_recovery_from_response(response))
        data = response.json()
        return DeviceAuthInitiation(
            device_code=data["device_code"],
            user_code=data["user_code"],
            verification_url=data["verification_uri"],
            verification_url_complete=data.get("verification_uri_complete"),
            expires_in=int(data["expires_in"]),
            interval=int(data.get("interval", 5)),
        )

    async def poll_for_tokens(self, device_code: str) -> DeviceAuthPollResult:
        client_id = microsoft_app().client_id
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{LOGIN_BASE}/common/oauth2/v2.0/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    **client_fields(client_id),
                },
            )
        if response.status_code == 200:
            credentials = await credentials_from_tokens(
                response.json(), client_id=client_id, auth_flow="device_code"
            )
            return DeviceAuthPollResult(status="approved", credentials=credentials)
        error = error_code(response)
        if error == "authorization_pending":
            return DeviceAuthPollResult(status="pending")
        if error == "slow_down":
            return DeviceAuthPollResult(status="slow_down", next_poll_interval=10)
        if error in {"authorization_declined", "access_denied"}:
            return DeviceAuthPollResult(status="denied")
        if error in {"expired_token", "bad_verification_code"}:
            return DeviceAuthPollResult(status="expired")
        raise MicrosoftAuthError(microsoft_recovery_from_response(response))

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        return await refresh_microsoft_credentials(credentials)

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        return False
