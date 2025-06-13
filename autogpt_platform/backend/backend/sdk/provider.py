"""
Provider configuration class that holds all provider-related settings.
"""

from typing import Any, Callable, List, Optional, Set, Type

from backend.data.cost import BlockCost
from backend.data.model import Credentials, CredentialsField, CredentialsMetaInput
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.integrations.webhooks._base import BaseWebhooksManager


class Provider:
    """A configured provider that blocks can use."""

    def __init__(
        self,
        name: str,
        oauth_handler: Optional[Type[BaseOAuthHandler]] = None,
        webhook_manager: Optional[Type[BaseWebhooksManager]] = None,
        default_credentials: Optional[List[Credentials]] = None,
        base_costs: Optional[List[BlockCost]] = None,
        supported_auth_types: Optional[Set[str]] = None,
        api_client_factory: Optional[Callable] = None,
        error_handler: Optional[Callable[[Exception], str]] = None,
        **kwargs,
    ):
        self.name = name
        self.oauth_handler = oauth_handler
        self.webhook_manager = webhook_manager
        self.default_credentials = default_credentials or []
        self.base_costs = base_costs or []
        self.supported_auth_types = supported_auth_types or set()
        self._api_client_factory = api_client_factory
        self._error_handler = error_handler

        # Store any additional configuration
        self._extra_config = kwargs

    def credentials_field(self, **kwargs) -> CredentialsMetaInput:
        """Return a CredentialsField configured for this provider."""
        # Merge provider defaults with user overrides
        field_kwargs = {
            "provider": self.name,
            "supported_credential_types": self.supported_auth_types,
            "description": f"{self.name.title()} credentials",
        }
        field_kwargs.update(kwargs)

        return CredentialsField(**field_kwargs)

    def get_api(self, credentials: Credentials) -> Any:
        """Get API client instance for the given credentials."""
        if self._api_client_factory:
            return self._api_client_factory(credentials)
        raise NotImplementedError(f"No API client factory registered for {self.name}")

    def handle_error(self, error: Exception) -> str:
        """Handle provider-specific errors."""
        if self._error_handler:
            return self._error_handler(error)
        return str(error)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get additional configuration value."""
        return self._extra_config.get(key, default)
