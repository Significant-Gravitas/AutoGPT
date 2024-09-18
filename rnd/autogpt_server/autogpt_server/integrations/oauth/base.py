import time
from abc import ABC, abstractmethod
from typing import ClassVar

from autogpt_libs.supabase_integration_credentials_store import OAuth2Credentials


class BaseOAuthHandler(ABC):
    PROVIDER_NAME: ClassVar[str]

    @abstractmethod
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str): ...

    @abstractmethod
    def get_login_url(self, scopes: list[str], state: str) -> str:
        """Constructs a login URL that the user can be redirected to"""
        ...

    @abstractmethod
    def exchange_code_for_tokens(self, code: str) -> OAuth2Credentials:
        """Exchanges the acquired authorization code from login for a set of tokens"""
        ...

    @abstractmethod
    def _refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
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
