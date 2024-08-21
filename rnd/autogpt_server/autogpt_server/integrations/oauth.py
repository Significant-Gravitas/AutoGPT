from abc import ABC, abstractmethod

from autogpt_libs.supabase_integration_credentials_store import OAuth2Credentials


class BaseOAuthHandler(ABC):
    @abstractmethod
    def get_login_url(self, scopes: list[str], state: str) -> str:
        ...

    @abstractmethod
    def exchange_code_for_tokens(self, code: str) -> OAuth2Credentials:
        ...

    @abstractmethod
    def get_auth_token(self, credentials: OAuth2Credentials) -> str:
        ...

    @abstractmethod
    def needs_refresh(self, credentials: OAuth2Credentials) -> bool:
        ...

    @abstractmethod
    def refresh_tokens(self, credentials: OAuth2Credentials) -> OAuth2Credentials:
        ...
