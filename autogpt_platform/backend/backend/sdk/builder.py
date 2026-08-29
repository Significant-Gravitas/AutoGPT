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
        self._description: Optional[str] = None
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

    def with_managed_api_key(self) -> "ProviderBuilder":
        """Declare api_key auth support without an env-var-backed default credential.

        Use for providers whose API key is provisioned per-user by a
        :class:`~backend.integrations.managed_credentials.ManagedCredentialProvider`
        (e.g. Ayrshare's profile key).  Equivalent to :meth:`with_api_key` but
        skips both the env-var lookup and the default-credential registration
        — nothing can accidentally leak an org-level key into blocks as if it
        were a per-user credential.
        """
        self._supported_auth_types.add("api_key")
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
        self,
        amount: int,
        cost_type: BlockCostType,
        cost_divisor: int = 1,
    ) -> "ProviderBuilder":
        """Set base cost for all blocks using this provider.

        ``cost_divisor`` only applies to SECOND / ITEMS. TOKENS is billed via
        the TOKEN_COST rate table (per-model pricing) and ignores the divisor.
        Example: ``with_base_cost(1, BlockCostType.SECOND, cost_divisor=10)``
        bills 1 credit per 10 walltime seconds.
        """
        self._base_costs.append(
            BlockCost(
                cost_amount=amount,
                cost_type=cost_type,
                cost_divisor=cost_divisor,
            )
        )
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

    def with_description(self, description: str) -> "ProviderBuilder":
        """Set a short human-readable description for this provider.

        Exposed via ``GET /integrations/providers`` so the frontend can display
        a one-line summary next to each provider without hardcoding copy
        (e.g. ``"Issues, PRs, repositories"`` for GitHub).
        """
        self._description = description
        return self

    def with_supported_auth_types(self, *types: CredentialsType) -> "ProviderBuilder":
        """Declare which credential types this provider accepts.

        Surfaced via ``GET /integrations/providers`` so the settings UI can
        render only the relevant connection tabs (OAuth / API key / etc.) per
        provider. ``with_oauth``, ``with_api_key``, ``with_managed_api_key``,
        and ``with_user_password`` populate this set automatically — call this
        method only for providers whose auth handlers/credentials live outside
        the builder chain (e.g. legacy OAuth handlers in
        ``backend/integrations/oauth/`` or block-level ``CredentialsMetaInput``
        declarations).
        """
        self._supported_auth_types.update(types)
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
            description=self._description,
            **self._extra_config,
        )

        # Auto-registration happens here
        AutoRegistry.register_provider(provider)
        return provider
