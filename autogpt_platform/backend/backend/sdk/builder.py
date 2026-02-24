"""
Builder class for creating provider configurations with a fluent API.
"""

import logging
import os
from typing import Callable, List, Optional, Type

from pydantic import SecretStr

from backend.blocks._base import BlockCost, BlockCostType
from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    CredentialsType,
    UserPasswordCredentials,
)
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.integrations.webhooks._base import BaseWebhooksManager
from backend.sdk.provider import OAuthConfig, Provider
from backend.sdk.registry import AutoRegistry
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


class ProviderBuilder:
    """Builder for creating provider configurations."""

    def __init__(self, name: str):
        self.name = name
        self._oauth_config: Optional[OAuthConfig] = None
        self._webhook_manager: Optional[Type[BaseWebhooksManager]] = None
        self._default_credentials: List[Credentials] = []
        self._base_costs: List[BlockCost] = []
        self._supported_auth_types: set[CredentialsType] = set()
        self._api_client_factory: Optional[Callable] = None
        self._error_handler: Optional[Callable[[Exception], str]] = None
        self._default_scopes: Optional[List[str]] = None
        self._client_id_env_var: Optional[str] = None
        self._client_secret_env_var: Optional[str] = None
        self._extra_config: dict = {}

    def with_oauth(
        self,
        handler_class: Type[BaseOAuthHandler],
        scopes: Optional[List[str]] = None,
        client_id_env_var: Optional[str] = None,
        client_secret_env_var: Optional[str] = None,
    ) -> "ProviderBuilder":
        """Add OAuth support."""
        if not client_id_env_var or not client_secret_env_var:
            client_id_env_var = f"{self.name}_client_id".upper()
            client_secret_env_var = f"{self.name}_client_secret".upper()

        if os.getenv(client_id_env_var) and os.getenv(client_secret_env_var):
            self._client_id_env_var = client_id_env_var
            self._client_secret_env_var = client_secret_env_var

            self._oauth_config = OAuthConfig(
                oauth_handler=handler_class,
                scopes=scopes,
                client_id_env_var=client_id_env_var,
                client_secret_env_var=client_secret_env_var,
            )
            self._supported_auth_types.add("oauth2")
        else:
            logger.warning(
                f"Provider {self.name.upper()} implements OAuth but the required env "
                f"vars {client_id_env_var} and {client_secret_env_var} are not both set"
            )
        return self

    def with_api_key(self, env_var_name: str, title: str) -> "ProviderBuilder":
        """Add API key support with environment variable name."""
        self._supported_auth_types.add("api_key")

        # Register the API key mapping
        AutoRegistry.register_api_key(self.name, env_var_name)

        # Check if API key exists in environment
        api_key = os.getenv(env_var_name)
        if api_key:
            self._default_credentials.append(
                APIKeyCredentials(
                    id=f"{self.name}-default",
                    provider=self.name,
                    api_key=SecretStr(api_key),
                    title=title,
                )
            )
        return self

    def with_api_key_from_settings(
        self, settings_attr: str, title: str
    ) -> "ProviderBuilder":
        """Use existing API key from settings."""
        self._supported_auth_types.add("api_key")

        # Try to get the API key from settings
        settings = Settings()
        api_key = getattr(settings.secrets, settings_attr, None)
        if api_key:
            self._default_credentials.append(
                APIKeyCredentials(
                    id=f"{self.name}-default",
                    provider=self.name,
                    api_key=api_key,
                    title=title,
                )
            )
        return self

    def with_user_password(
        self, username_env_var: str, password_env_var: str, title: str
    ) -> "ProviderBuilder":
        """Add username/password support with environment variable names."""
        self._supported_auth_types.add("user_password")

        # Check if credentials exist in environment
        username = os.getenv(username_env_var)
        password = os.getenv(password_env_var)
        if username and password:
            self._default_credentials.append(
                UserPasswordCredentials(
                    id=f"{self.name}-default",
                    provider=self.name,
                    username=SecretStr(username),
                    password=SecretStr(password),
                    title=title,
                )
            )
        return self

    def with_webhook_manager(
        self, manager_class: Type[BaseWebhooksManager]
    ) -> "ProviderBuilder":
        """Register webhook manager for this provider."""
        self._webhook_manager = manager_class
        return self

    def with_base_cost(
        self, amount: int, cost_type: BlockCostType
    ) -> "ProviderBuilder":
        """Set base cost for all blocks using this provider."""
        self._base_costs.append(BlockCost(cost_amount=amount, cost_type=cost_type))
        return self

    def with_api_client(self, factory: Callable) -> "ProviderBuilder":
        """Register API client factory."""
        self._api_client_factory = factory
        return self

    def with_error_handler(
        self, handler: Callable[[Exception], str]
    ) -> "ProviderBuilder":
        """Register error handler for provider-specific errors."""
        self._error_handler = handler
        return self

    def with_config(self, **kwargs) -> "ProviderBuilder":
        """Add additional configuration options."""
        self._extra_config.update(kwargs)
        return self

    def build(self) -> Provider:
        """Build and register the provider configuration."""
        provider = Provider(
            name=self.name,
            oauth_config=self._oauth_config,
            webhook_manager=self._webhook_manager,
            default_credentials=self._default_credentials,
            base_costs=self._base_costs,
            supported_auth_types=self._supported_auth_types,
            api_client_factory=self._api_client_factory,
            error_handler=self._error_handler,
            **self._extra_config,
        )

        # Auto-registration happens here
        AutoRegistry.register_provider(provider)
        return provider
