"""
Stripe Link — OAuth 2.0 Device Code Grant handler.

Implements the device code flow for Stripe Link (login.link.com).
Uses a public client ID (no client_secret).
"""

import logging
import time
from typing import ClassVar
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.device_base import (
    BaseDeviceAuthHandler,
    DeviceAuthInitiation,
    DeviceAuthPollResult,
)
from backend.integrations.providers import ProviderName
from backend.util.settings import Config

logger = logging.getLogger(__name__)
app_config = Config()

LINK_AUTH_BASE_URL = "https://login.link.com"
LINK_CLIENT_ID = "lwlpk_U7Qy7ThG69STZk"
LINK_CLIENT_NAME = "AutoGPT"


class StripeLinkDeviceAuthHandler(BaseDeviceAuthHandler):
    """Device code handler for Stripe Link."""

    PROVIDER_NAME: ClassVar[ProviderName | str] = ProviderName.STRIPE_LINK
    DEFAULT_SCOPES: ClassVar[list[str]] = [
        "userinfo:read",
        "payment_methods.agentic",
    ]
    # Requested as RFC 9396 authorization details, not scopes — Link gates its
    # read endpoints on these rather than on the scope string.
    SOURCE_ACTIONS: ClassVar[list[str]] = [
        "read_balances",
        "read_external_transactions",
        "read_link_transactions",
        "read_source_details",
    ]

    async def initiate_device_auth(self, scopes: list[str]) -> DeviceAuthInitiation:
        effective_scopes = self.handle_default_scopes(scopes)

        # Shown in the user's Link app under connected apps, and on the consent
        # screen. The CLI uses `<client> on <hostname>`, but our hostname is a
        # container ID the user has never seen — the platform they are actually
        # connecting to is the useful half.
        platform_host = urlparse(app_config.platform_base_url or "").netloc
        connection_label = (
            f"{LINK_CLIENT_NAME} on {platform_host}"
            if platform_host
            else LINK_CLIENT_NAME
        )

        # RFC 9396 rich authorization details. Link gates its read endpoints
        # (/balances, /transactions, /sources) on these source actions being
        # part of the grant — without them those return 403 feature_unavailable
        # even though the scopes look sufficient. One `source` detail carries
        # every action, matching the CLI's buildAuthorizationDetails.
        form = {
            "client_id": LINK_CLIENT_ID,
            "scope": " ".join(effective_scopes),
            "connection_label": connection_label,
            "client_hint": LINK_CLIENT_NAME,
            "authorization_details[][type]": "source",
            "authorization_details[][actions][]": self.SOURCE_ACTIONS,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LINK_AUTH_BASE_URL}/device/code",
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LINK_AUTH_BASE_URL}/device/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": LINK_CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if response.status_code == 200:
            data = response.json()
            credentials = OAuth2Credentials(
                provider=self.PROVIDER_NAME,
                access_token=SecretStr(data["access_token"]),
                refresh_token=SecretStr(data["refresh_token"]),
                access_token_expires_at=int(time.time()) + data["expires_in"],
                scopes=self.DEFAULT_SCOPES,
                title="Stripe Link",
            )
            return DeviceAuthPollResult(status="approved", credentials=credentials)

        if response.status_code == 400:
            error = response.json()
            error_code = error.get("error", "")

            if error_code == "authorization_pending":
                return DeviceAuthPollResult(status="pending")

            if error_code == "slow_down":
                return DeviceAuthPollResult(
                    status="slow_down",
                    next_poll_interval=10,
                )

            if error_code == "expired_token":
                return DeviceAuthPollResult(status="expired")

            if error_code == "access_denied":
                return DeviceAuthPollResult(status="denied")

        raise RuntimeError(
            f"Unexpected response from Link auth: "
            f"{response.status_code} {response.text}"
        )

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise RuntimeError("No refresh token available")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LINK_AUTH_BASE_URL}/device/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": credentials.refresh_token.get_secret_value(),
                    "client_id": LINK_CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()

        credentials.access_token = SecretStr(data["access_token"])
        credentials.refresh_token = SecretStr(data["refresh_token"])
        credentials.access_token_expires_at = int(time.time()) + data["expires_in"]
        return credentials

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not credentials.refresh_token:
            return False

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LINK_AUTH_BASE_URL}/device/revoke",
                data={
                    "client_id": LINK_CLIENT_ID,
                    "token": credentials.refresh_token.get_secret_value(),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        return response.status_code == 200
