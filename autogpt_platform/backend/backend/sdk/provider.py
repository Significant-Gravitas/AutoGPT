"""
Provider configuration class that holds all provider-related settings.
"""

from typing import Any, Callable, List, Optional, Set, Type

from pydantic import BaseModel

from backend.data.cost import BlockCost
from backend.data.model import Credentials, CredentialsField, CredentialsMetaInput
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.integrations.webhooks._base import BaseWebhooksManager


class OAuthConfig(BaseModel):
    """Configuration for OAuth authentication."""

    oauth_handler: Type[BaseOAuthHandler]
    scopes: Optional[List[str]] = None
    client_id_env_var: Optional[str] = None
    client_secret_env_var: Optional[str] = None


class Provider:
    """A configured provider that blocks can use.

    A Provider represents a service or platform that blocks can integrate with, like Linear, OpenAI, etc.
    It contains configuration for:
    - Authentication (OAuth, API keys)
    - Default credentials
    - Base costs for using the provider
    - Webhook handling
    - Error handling
    - API client factory

    Blocks use Provider instances to handle authentication, make API calls, and manage service-specific logic.
    """

    def __init__(
        self,
        name: str,
        oauth_config: Optional[OAuthConfig] = None,
        webhook_manager: Optional[Type[BaseWebhooksManager]] = None,
        default_credentials: Optional[List[Credentials]] = None,
        base_costs: Optional[List[BlockCost]] = None,
        supported_auth_types: Optional[Set[str]] = None,
        api_client_factory: Optional[Callable] = None,
        error_handler: Optional[Callable[[Exception], str]] = None,
        **kwargs,
    ):
        self.name = name
        self.oauth_config = oauth_config
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
        # Extract known CredentialsField parameters
        title = kwargs.pop("title", None)
        description = kwargs.pop("description", f"{self.name.title()} credentials")
        required_scopes = kwargs.pop("required_scopes", set())
        discriminator = kwargs.pop("discriminator", None)
        discriminator_mapping = kwargs.pop("discriminator_mapping", None)
        discriminator_values = kwargs.pop("discriminator_values", None)

        # Create json_schema_extra with provider information
        json_schema_extra = {
            "credentials_provider": [self.name],
            "credentials_types": (
                list(self.supported_auth_types)
                if self.supported_auth_types
                else ["api_key"]
            ),
        }

        # Merge any existing json_schema_extra
        if "json_schema_extra" in kwargs:
            json_schema_extra.update(kwargs.pop("json_schema_extra"))

        # Add json_schema_extra to kwargs
        kwargs["json_schema_extra"] = json_schema_extra

        return CredentialsField(
            required_scopes=required_scopes,
            discriminator=discriminator,
            discriminator_mapping=discriminator_mapping,
            discriminator_values=discriminator_values,
            title=title,
            description=description,
            **kwargs,
        )

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
