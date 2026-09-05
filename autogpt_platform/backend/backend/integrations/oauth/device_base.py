"""
Base handler for OAuth 2.0 Device Code Grant (RFC 8628).

Providers that use the device authorization flow (CLI tools, IoT devices,
smart TVs, etc.) implement this handler instead of ``BaseOAuthHandler``.

The resulting credentials are standard ``OAuth2Credentials`` — the device
code flow is only an *acquisition method*, not a different credential shape.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel

from backend.data.model import OAuth2Credentials
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)


class DeviceAuthInitiation(BaseModel):
    """Returned when initiating the device code flow."""

    device_code: str
    user_code: str
    verification_url: str
    verification_url_complete: Optional[str] = None
    expires_in: int
    interval: int  # recommended seconds between polls


class DeviceAuthPollResult(BaseModel):
    """Returned from each poll attempt during the device code flow."""

    status: Literal["pending", "slow_down", "approved", "denied", "expired"]
    credentials: Optional[OAuth2Credentials] = None
    next_poll_interval: Optional[int] = None


class BaseDeviceAuthHandler(ABC):
    """Base for RFC 8628 device-code handlers.

    ``ROTATES_REFRESH_TOKEN`` tells the credentials manager it must serialize
    refreshes for this provider: replaying a rotated refresh token can be
    treated as compromise and revoke the whole grant.
    """

    ROTATES_REFRESH_TOKEN: ClassVar[bool] = False
    """
    Abstract handler for OAuth 2.0 Device Code Grant flows.

    Subclasses implement provider-specific HTTP calls; the token lifecycle
    helpers (refresh, needs_refresh, get_access_token) mirror
    ``BaseOAuthHandler`` so the credential manager can dispatch refresh
    calls uniformly.
    """

    PROVIDER_NAME: ClassVar[ProviderName | str]
    DEFAULT_SCOPES: ClassVar[list[str]] = []

    @abstractmethod
    async def initiate_device_auth(self, scopes: list[str]) -> DeviceAuthInitiation:
        """Start the device code flow. Returns URLs/codes for the user."""
        ...

    @abstractmethod
    async def poll_for_tokens(self, device_code: str) -> DeviceAuthPollResult:
        """
        Poll the auth server for token completion.

        Returns a result with status ``"pending"`` or ``"slow_down"`` while
        waiting, ``"approved"`` with credentials on success, or
        ``"denied"``/``"expired"`` on terminal failure.
        """
        ...

    @abstractmethod
    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        """Implements the token refresh mechanism."""
        ...

    @abstractmethod
    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        """Revokes the given token at the provider.
        Returns False if the provider does not support revocation."""
        ...

    # ------------------------------------------------------------------ #
    # Non-abstract helpers — same interface as BaseOAuthHandler
    # ------------------------------------------------------------------ #

    async def refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        if credentials.provider != self.PROVIDER_NAME:
            raise ValueError(
                f"{self.__class__.__name__} cannot refresh tokens "
                f"for provider '{credentials.provider}'"
            )
        return await self._refresh_tokens(credentials)

    async def get_access_token(self, credentials: OAuth2Credentials) -> str:
        """Returns a valid access token, refreshing it first if needed."""
        if self.needs_refresh(credentials):
            credentials = await self.refresh_tokens(credentials)
        return credentials.access_token.get_secret_value()

    def needs_refresh(self, credentials: OAuth2Credentials) -> bool:
        """Indicates whether the given tokens need to be refreshed."""
        return (
            credentials.access_token_expires_at is not None
            and credentials.access_token_expires_at < int(time.time()) + 300
        )

    def handle_default_scopes(self, scopes: list[str]) -> list[str]:
        """Uses default scopes when none are provided."""
        if not scopes:
            logger.debug("Using default scopes for provider %s", self.PROVIDER_NAME)
            scopes = self.DEFAULT_SCOPES
        return scopes
