import logging
import time
from abc import ABC, abstractmethod
from typing import ClassVar

from autogpt_libs.supabase_integration_credentials_store import OAuth2Credentials

logger = logging.getLogger(__name__)


class BaseOAuthHandler(ABC):
    # --8<-- [start:BaseOAuthHandler1]
    PROVIDER_NAME: ClassVar[str]
    DEFAULT_SCOPES: ClassVar[list[str]] = []
    # --8<-- [end:BaseOAuthHandler1]

    @abstractmethod
    # --8<-- [start:BaseOAuthHandler2]
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str): ...

    # --8<-- [end:BaseOAuthHandler2]

    @abstractmethod
    # --8<-- [start:BaseOAuthHandler3]
    def get_login_url(self, scopes: list[str], state: str) -> str:
        # --8<-- [end:BaseOAuthHandler3]
        """Constructs a login URL that the user can be redirected to"""
        ...

    @abstractmethod
    # --8<-- [start:BaseOAuthHandler4]
    def exchange_code_for_tokens(
        self, code: str, scopes: list[str]
    ) -> OAuth2Credentials:
        # --8<-- [end:BaseOAuthHandler4]
        """Exchanges the acquired authorization code from login for a set of tokens"""
        ...

    @abstractmethod
    # --8<-- [start:BaseOAuthHandler5]
    def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        # --8<-- [end:BaseOAuthHandler5]
        """Implements the token refresh mechanism"""
        ...

    def refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        if credentials.provider != self.PROVIDER_NAME:
            raise ValueError(
                f"{self.__class__.__name__} can not refresh tokens "
                f"for other provider '{credentials.provider}'"
            )
        return self._refresh_tokens(credentials)

    def get_access_token(self, credentials: OAuth2Credentials) -> str:
        """Returns a valid access token, refreshing it first if needed"""
        if self.needs_refresh(credentials):
            credentials = self.refresh_tokens(credentials)
        return credentials.access_token.get_secret_value()

    def needs_refresh(self, credentials: OAuth2Credentials) -> bool:
        """Indicates whether the given tokens need to be refreshed"""
        return (
            credentials.access_token_expires_at is not None
            and credentials.access_token_expires_at < int(time.time()) + 300
        )

    def handle_default_scopes(self, scopes: list[str]) -> list[str]:
        """Handles the default scopes for the provider"""
        # If scopes are empty, use the default scopes for the provider
        if not scopes:
            logger.debug(f"Using default scopes for provider {self.PROVIDER_NAME}")
            scopes = self.DEFAULT_SCOPES
        return scopes
