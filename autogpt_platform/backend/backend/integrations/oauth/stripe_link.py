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
LINK_API_BASE_URL = "https://api.link.com"

# Every Link call is awaited inline by a block or an API handler, so an
# upstream that accepts the connection and then stalls would hold a worker for
# as long as it likes. Bound them all.
LINK_HTTP_TIMEOUT = 15.0
LINK_CLIENT_ID = "lwlpk_U7Qy7ThG69STZk"
LINK_CLIENT_NAME = "AutoGPT"


# Link truncates `connection_label` at 32 characters, and the label is the
# headline of the approval sheet ("<label> is requesting to spend $X"), so an
# overlong host renders as e.g. "AutoGPT on prompt-neat-flea.ngro".
CONNECTION_LABEL_MAX_LEN = 32


def _connection_label(platform_host: str) -> str:
    """Build the approval-sheet label, dropping the host if it will not fit."""
    if not platform_host:
        return LINK_CLIENT_NAME
    label = f"{LINK_CLIENT_NAME} on {platform_host}"
    if len(label) <= CONNECTION_LABEL_MAX_LEN:
        return label

    # A truncated host is worse than none: better a clean "AutoGPT" than
    # "AutoGPT on prompt-neat-flea.ngro".
    #
    # Falling back to the last two labels was tried and reverted. It reads well
    # for a corporate host (agpt.internal.mycompany.example -> mycompany.example)
    # but not for shared hosting: prompt-neat-flea.ngrok-free.app becomes
    # "AutoGPT on ngrok-free.app", which every ngrok user would also see. On the
    # sheet where someone decides whether to hand over money, a label that
    # implies an identity it does not carry is worse than a generic one, and
    # telling the two cases apart needs a public suffix list.
    return LINK_CLIENT_NAME


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
        connection_label = _connection_label(platform_host)

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

        async with httpx.AsyncClient(timeout=LINK_HTTP_TIMEOUT) as client:
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
        async with httpx.AsyncClient(timeout=LINK_HTTP_TIMEOUT) as client:
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
            # Record what was actually granted, not what was asked for. The
            # two can differ, and storing the request means a later scope
            # check passes on scopes the token may not carry. Link returns a
            # space-delimited `scope` string per RFC 6749; fall back to the
            # requested set only when it is absent.
            granted = str(data.get("scope") or "").split()
            credentials = OAuth2Credentials(
                provider=self.PROVIDER_NAME,
                access_token=SecretStr(data["access_token"]),
                refresh_token=SecretStr(data["refresh_token"]),
                access_token_expires_at=int(time.time()) + data["expires_in"],
                # No `or DEFAULT_SCOPES` fallback: recording the *requested*
                # scopes as granted is exactly what this must not do. A
                # credential that claims a scope its token lacks passes the
                # up-front coverage check and then 403s at run time, and an
                # inflated list can win `_merge_or_create_credential`'s
                # superset check and overwrite a wider credential.
                scopes=granted,
                title="Stripe Link",
                # Lets `_merge_or_create_credential` recognise a re-auth of the
                # same wallet instead of stacking a second credential for it.
                username=await self._fetch_username(data["access_token"]),
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

    async def _fetch_username(self, access_token: str) -> str | None:
        """Best-effort wallet identity, used only to de-duplicate credentials.

        A failure here must not fail an otherwise-successful authorization:
        the grant is already complete by this point, and the caller would be
        left with an approved device code and no credential.
        """
        try:
            async with httpx.AsyncClient(timeout=LINK_HTTP_TIMEOUT) as client:
                response = await client.get(
                    f"{LINK_API_BASE_URL}/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
            if response.status_code != 200:
                return None
            info = response.json()
            return info.get("email") or info.get("phone") or None
        except Exception as e:
            logger.warning(f"Could not read Link userinfo for credential title: {e}")
            return None

    # Link *rotates* the refresh token, unlike every other handler here. The
    # credentials manager documents concurrent refreshes as tolerable because
    # "the last writer wins and stale tokens are overwritten" — true only for a
    # static refresh token. Two workers replaying the same rotated token can
    # have the whole grant revoked by a provider that treats reuse as
    # compromise, so refreshes for this provider must be serialized.
    ROTATES_REFRESH_TOKEN: ClassVar[bool] = True

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise RuntimeError("No refresh token available")

        async with httpx.AsyncClient(timeout=LINK_HTTP_TIMEOUT) as client:
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
        # RFC 6749 §6 makes `refresh_token` optional on refresh; a provider that
        # does not rotate simply omits it. Indexing it unguarded would raise
        # inside `_refresh_locked`, leaving the credential un-updated and every
        # later run failing the same way with nothing pointing at the cause.
        if data.get("refresh_token"):
            credentials.refresh_token = SecretStr(data["refresh_token"])
        if data.get("expires_in") is not None:
            credentials.access_token_expires_at = int(time.time()) + int(
                data["expires_in"]
            )
        return credentials

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        if not credentials.refresh_token:
            return False

        async with httpx.AsyncClient(timeout=LINK_HTTP_TIMEOUT) as client:
            response = await client.post(
                f"{LINK_AUTH_BASE_URL}/device/revoke",
                data={
                    "client_id": LINK_CLIENT_ID,
                    "token": credentials.refresh_token.get_secret_value(),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        return response.status_code == 200
