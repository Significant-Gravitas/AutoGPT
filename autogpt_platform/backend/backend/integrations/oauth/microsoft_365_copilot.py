import logging
import time
from typing import ClassVar

import httpx
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.oauth.device_base import (
    BaseDeviceAuthHandler,
    DeviceAuthInitiation,
    DeviceAuthPollResult,
)
from backend.integrations.providers import ProviderName
from backend.util.settings import Settings

MICROSOFT_365_COPILOT_PUBLIC_CLIENT_ID = "ce01a4d6-be71-42a0-86b1-6540838eb17c"
MICROSOFT_DEVICE_CODE_URL = (
    "https://login.microsoftonline.com/organizations/oauth2/v2.0/devicecode"
)
MICROSOFT_TOKEN_URL = (
    "https://login.microsoftonline.com/organizations/oauth2/v2.0/token"
)
MICROSOFT_PROFILE_URL = "https://graph.microsoft.com/v1.0/me"
MICROSOFT_AUTH_TIMEOUT_SECONDS = 20.0

logger = logging.getLogger(__name__)


def _client_id() -> str:
    return (
        Settings().secrets.microsoft_client_id or MICROSOFT_365_COPILOT_PUBLIC_CLIENT_ID
    )


class Microsoft365CopilotDeviceAuthHandler(BaseDeviceAuthHandler):
    PROVIDER_NAME: ClassVar[ProviderName | str] = ProviderName.MICROSOFT_365_COPILOT
    CHAT_SCOPES: ClassVar[list[str]] = [
        "Sites.Read.All",
        "Mail.Read",
        "People.Read.All",
        "OnlineMeetingTranscript.Read.All",
        "Chat.Read",
        "ChannelMessage.Read.All",
        "ExternalItem.Read.All",
    ]
    DEFAULT_SCOPES: ClassVar[list[str]] = ["User.Read", *CHAT_SCOPES]

    async def initiate_device_auth(self, scopes: list[str]) -> DeviceAuthInitiation:
        effective_scopes = self._authorization_scopes(scopes)
        async with httpx.AsyncClient(timeout=MICROSOFT_AUTH_TIMEOUT_SECONDS) as client:
            response = await client.post(
                MICROSOFT_DEVICE_CODE_URL,
                data={
                    "client_id": _client_id(),
                    "scope": " ".join(effective_scopes),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        response.raise_for_status()
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
        async with httpx.AsyncClient(timeout=MICROSOFT_AUTH_TIMEOUT_SECONDS) as client:
            response = await client.post(
                MICROSOFT_TOKEN_URL,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": _client_id(),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if response.status_code == 200:
                credentials = self._credentials_from_token_response(response.json())
                await self._populate_account(client, credentials)
                return DeviceAuthPollResult(status="approved", credentials=credentials)

        error_code = self._error_code(response)
        if error_code == "authorization_pending":
            return DeviceAuthPollResult(status="pending")
        if error_code == "slow_down":
            return DeviceAuthPollResult(
                status="slow_down",
                next_poll_interval=10,
            )
        if error_code in {"authorization_declined", "access_denied"}:
            return DeviceAuthPollResult(status="denied")
        if error_code in {"expired_token", "bad_verification_code"}:
            return DeviceAuthPollResult(status="expired")
        raise RuntimeError(
            "Microsoft device token request failed "
            f"(status={response.status_code}, code={error_code or 'unknown'})"
        )

    async def _populate_account(
        self,
        client: httpx.AsyncClient,
        credentials: OAuth2Credentials,
    ) -> None:
        try:
            response = await client.get(
                MICROSOFT_PROFILE_URL,
                params={"$select": "displayName,mail,userPrincipalName,id"},
                headers={"Authorization": credentials.auth_header()},
            )
            response.raise_for_status()
            profile = response.json()
        except Exception as error:
            logger.warning(
                "Could not resolve Microsoft 365 Copilot account profile: %s",
                type(error).__name__,
            )
            return

        if not isinstance(profile, dict):
            return
        username = profile.get("mail") or profile.get("userPrincipalName")
        display_name = profile.get("displayName")
        account_id = profile.get("id")
        if isinstance(username, str) and username:
            credentials.username = username
        if isinstance(display_name, str) and display_name:
            credentials.title = display_name
        if isinstance(account_id, str) and account_id:
            credentials.metadata["microsoft_account_id"] = account_id

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if not credentials.refresh_token:
            raise ValueError("Microsoft 365 Copilot credentials have no refresh token")
        async with httpx.AsyncClient(timeout=MICROSOFT_AUTH_TIMEOUT_SECONDS) as client:
            response = await client.post(
                MICROSOFT_TOKEN_URL,
                data={
                    "client_id": _client_id(),
                    "refresh_token": credentials.refresh_token.get_secret_value(),
                    "grant_type": "refresh_token",
                    "scope": " ".join(self._authorization_scopes(credentials.scopes)),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        response.raise_for_status()
        return self._credentials_from_token_response(response.json(), credentials)

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        return False

    def _authorization_scopes(self, scopes: list[str]) -> list[str]:
        effective = self.handle_default_scopes(scopes)
        return list(dict.fromkeys([*effective, "offline_access"]))

    @staticmethod
    def _error_code(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return ""
        return str(data.get("error") or "") if isinstance(data, dict) else ""

    def _credentials_from_token_response(
        self,
        token_data: dict,
        current: OAuth2Credentials | None = None,
    ) -> OAuth2Credentials:
        access_token = token_data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("Microsoft token response did not include an access token")

        refresh_token = token_data.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            refresh_token = (
                current.refresh_token.get_secret_value()
                if current and current.refresh_token
                else None
            )
        if not refresh_token:
            raise ValueError("Microsoft token response did not include a refresh token")

        raw_scopes = token_data.get("scope", "")
        scopes = (
            [scope.rsplit("/", 1)[-1] for scope in raw_scopes.split()]
            if isinstance(raw_scopes, str)
            else []
        )
        if not scopes:
            scopes = list(current.scopes if current else self.DEFAULT_SCOPES)

        expires_in = token_data.get("expires_in")
        try:
            expires_at = (
                int(time.time()) + int(expires_in)
                if isinstance(expires_in, (int, str))
                else None
            )
        except ValueError:
            expires_at = None
        credentials = OAuth2Credentials(
            provider=self.PROVIDER_NAME,
            title=current.title if current else "Microsoft 365 Copilot",
            username=current.username if current else None,
            access_token=SecretStr(access_token),
            refresh_token=SecretStr(refresh_token),
            access_token_expires_at=expires_at,
            refresh_token_expires_at=None,
            scopes=scopes,
        )
        if current:
            credentials.id = current.id
        return credentials
