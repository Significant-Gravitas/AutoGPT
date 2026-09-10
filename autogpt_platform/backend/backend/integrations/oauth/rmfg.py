"""
RMFG — OAuth 2.0 Device Code Grant handler (RFC 8628).

RMFG issues no client secret: the public client id ``rmfg-agent`` starts a
device authorization, the person approves it at rmfg.com/connect, and the
token endpoint returns an access token plus a *rotating* refresh token.
Access tokens last up to 15 minutes; a connection lasts 30 days.
"""

import logging
import time
from typing import Any, ClassVar

import httpx
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.device_base import (
    BaseDeviceAuthHandler,
    DeviceAuthInitiation,
    DeviceAuthPollResult,
)
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)

RMFG_API_URL = "https://api.rmfg.com"
RMFG_OAUTH_URL = f"{RMFG_API_URL}/v1/oauth"
# Every call here is awaited inline by an API handler or the credentials
# manager, so a stalled upstream must not hold a worker indefinitely.
RMFG_HTTP_TIMEOUT = 15.0
RMFG_CLIENT_ID = "rmfg-agent"
DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
FORM_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}

# Granted to every connection; nearly every RMFG block needs them.
BASE_SCOPES = ["designs", "dfm", "quotes", "carts", "orders"]
# Shown as opt-in choices on RMFG's approval page: ``webhooks`` for the
# trigger block, ``payments`` for Pay Cart. A person can decline either while
# the connection still succeeds, so the granted set is read from the token.
OPTIONAL_SCOPES = ["webhooks", "payments"]


class RMFGDeviceAuthHandler(BaseDeviceAuthHandler):
    """Device code handler for RMFG."""

    PROVIDER_NAME: ClassVar[ProviderName | str] = ProviderName.RMFG
    DEFAULT_SCOPES: ClassVar[list[str]] = BASE_SCOPES + OPTIONAL_SCOPES
    # RMFG treats reuse of a consumed refresh token as compromise and revokes
    # the whole connection, so refreshes must be serialized.
    ROTATES_REFRESH_TOKEN: ClassVar[bool] = True

    async def initiate_device_auth(self, scopes: list[str]) -> DeviceAuthInitiation:
        requested = self.handle_default_scopes(scopes)
        # A block that needs one optional permission asks for just that scope.
        # The base set has to ride along, or the resulting token could not
        # even read the design the block is acting on.
        effective = BASE_SCOPES + [s for s in requested if s not in BASE_SCOPES]

        async with httpx.AsyncClient(timeout=RMFG_HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{RMFG_OAUTH_URL}/device/code",
                data={"client_id": RMFG_CLIENT_ID, "scope": " ".join(effective)},
                headers=FORM_HEADERS,
            )
            response.raise_for_status()
            data = response.json()

        return DeviceAuthInitiation(
            device_code=data["device_code"],
            user_code=data["user_code"],
            verification_url=data["verification_uri"],
            verification_url_complete=data.get("verification_uri_complete"),
            expires_in=data["expires_in"],
            interval=data.get("interval", 5),
        )

    async def poll_for_tokens(self, device_code: str) -> DeviceAuthPollResult:
        async with httpx.AsyncClient(timeout=RMFG_HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{RMFG_OAUTH_URL}/token",
                data={
                    "grant_type": DEVICE_CODE_GRANT,
                    "client_id": RMFG_CLIENT_ID,
                    "device_code": device_code,
                },
                headers=FORM_HEADERS,
            )

        if response.status_code == 200:
            data = response.json()
            credentials = OAuth2Credentials(
                provider=self.PROVIDER_NAME,
                access_token=SecretStr(data["access_token"]),
                refresh_token=SecretStr(data["refresh_token"]),
                access_token_expires_at=int(time.time()) + int(data["expires_in"]),
                # Record what was granted, not what was asked for: the person
                # may have left "Also allow paid orders" unchecked.
                scopes=str(data.get("scope") or "").split(),
                title="RMFG",
                # Lets a re-connect of the same account update the existing
                # credential instead of stacking a second one.
                username=await self._fetch_username(data["access_token"]),
            )
            return DeviceAuthPollResult(status="approved", credentials=credentials)

        # RFC 8628 reports every waiting and terminal state as HTTP 400.
        if response.status_code == 400:
            error_code = response.json().get("error", "")
            if error_code == "authorization_pending":
                return DeviceAuthPollResult(status="pending")
            if error_code == "slow_down":
                # RMFG asks for the interval plus five seconds.
                return DeviceAuthPollResult(status="slow_down", next_poll_interval=10)
            if error_code == "expired_token":
                return DeviceAuthPollResult(status="expired")
            if error_code == "access_denied":
                return DeviceAuthPollResult(status="denied")

        raise RuntimeError(
            f"Unexpected response from RMFG auth: "
            f"{response.status_code} {response.text[:200]}"
        )

    async def _fetch_username(self, access_token: str) -> str | None:
        """Best-effort account email, used only to de-duplicate credentials.

        The grant is already complete here; a failure must not turn an
        approved authorization into a lost credential.
        """
        try:
            async with httpx.AsyncClient(timeout=RMFG_HTTP_TIMEOUT) as client:
                response = await client.get(
                    f"{RMFG_API_URL}/v1/account",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
            if response.status_code != 200:
                return None
            email = response.json().get("email")
            return str(email) if email else None
        except Exception as exc:
            logger.warning(f"Could not read RMFG account for credential title: {exc}")
            return None

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise RuntimeError("No refresh token available")

        async with httpx.AsyncClient(timeout=RMFG_HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{RMFG_OAUTH_URL}/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": RMFG_CLIENT_ID,
                    "refresh_token": credentials.refresh_token.get_secret_value(),
                },
                headers=FORM_HEADERS,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        credentials.access_token = SecretStr(data["access_token"])
        # RMFG rotates the refresh token on every refresh and revokes the
        # connection if a consumed one is replayed, so the replacement must be
        # saved every time. RFC 6749 still makes the field optional.
        if data.get("refresh_token"):
            credentials.refresh_token = SecretStr(data["refresh_token"])
        if data.get("expires_in") is not None:
            credentials.access_token_expires_at = int(time.time()) + int(
                data["expires_in"]
            )
        if data.get("scope"):
            credentials.scopes = str(data["scope"]).split()
        return credentials

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not credentials.refresh_token:
            return False

        async with httpx.AsyncClient(timeout=RMFG_HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{RMFG_OAUTH_URL}/revoke",
                data={
                    "client_id": RMFG_CLIENT_ID,
                    "token": credentials.refresh_token.get_secret_value(),
                },
                headers=FORM_HEADERS,
            )

        return response.status_code == 200
