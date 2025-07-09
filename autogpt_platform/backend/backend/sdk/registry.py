"""
Auto-registration system for blocks, providers, and their configurations.
"""

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from pydantic import BaseModel, SecretStr

from backend.blocks.basic import Block
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.integrations.webhooks._base import BaseWebhooksManager

if TYPE_CHECKING:
    from backend.sdk.provider import Provider


class SDKOAuthCredentials(BaseModel):
    """OAuth credentials configuration for SDK providers."""

    use_secrets: bool = False
    client_id_env_var: Optional[str] = None
    client_secret_env_var: Optional[str] = None


class BlockConfiguration:
    """Configuration associated with a block."""

    def __init__(
        self,
        provider: str,
        costs: List[Any],
        default_credentials: List[Credentials],
        webhook_manager: Optional[Type[BaseWebhooksManager]] = None,
        oauth_handler: Optional[Type[BaseOAuthHandler]] = None,
    ):
        self.provider = provider
        self.costs = costs
        self.default_credentials = default_credentials
        self.webhook_manager = webhook_manager
        self.oauth_handler = oauth_handler


class AutoRegistry:
    """Central registry for all block-related configurations."""

    _lock = threading.Lock()
    _providers: Dict[str, "Provider"] = {}
    _default_credentials: List[Credentials] = []
    _oauth_handlers: Dict[str, Type[BaseOAuthHandler]] = {}
    _oauth_credentials: Dict[str, SDKOAuthCredentials] = {}
    _webhook_managers: Dict[str, Type[BaseWebhooksManager]] = {}
    _block_configurations: Dict[Type[Block], BlockConfiguration] = {}
    _api_key_mappings: Dict[str, str] = {}  # provider -> env_var_name

    @classmethod
    def register_provider(cls, provider: "Provider") -> None:
        """Auto-register provider and all its configurations."""
        with cls._lock:
            cls._providers[provider.name] = provider

            # Register OAuth handler if provided
            if provider.oauth_config:
                # Dynamically set PROVIDER_NAME if not already set
                if (
                    not hasattr(provider.oauth_config.oauth_handler, "PROVIDER_NAME")
                    or provider.oauth_config.oauth_handler.PROVIDER_NAME is None
                ):
                    # Import ProviderName to create dynamic enum value
                    from backend.integrations.providers import ProviderName

                    # This works because ProviderName has _missing_ method
                    provider.oauth_config.oauth_handler.PROVIDER_NAME = ProviderName(
                        provider.name
                    )
                cls._oauth_handlers[provider.name] = provider.oauth_config.oauth_handler

                # Register OAuth credentials configuration
                oauth_creds = SDKOAuthCredentials(
                    use_secrets=False,  # SDK providers use custom env vars
                    client_id_env_var=provider.oauth_config.client_id_env_var,
                    client_secret_env_var=provider.oauth_config.client_secret_env_var,
                )
                cls._oauth_credentials[provider.name] = oauth_creds

            # Register webhook manager if provided
            if provider.webhook_manager:
                # Dynamically set PROVIDER_NAME if not already set
                if (
                    not hasattr(provider.webhook_manager, "PROVIDER_NAME")
                    or provider.webhook_manager.PROVIDER_NAME is None
                ):
                    # Import ProviderName to create dynamic enum value
                    from backend.integrations.providers import ProviderName

                    # This works because ProviderName has _missing_ method
                    provider.webhook_manager.PROVIDER_NAME = ProviderName(provider.name)
                cls._webhook_managers[provider.name] = provider.webhook_manager

            # Register default credentials
            cls._default_credentials.extend(provider.default_credentials)

    @classmethod
    def register_api_key(cls, provider: str, env_var_name: str) -> None:
        """Register an environment variable as an API key for a provider."""
        with cls._lock:
            cls._api_key_mappings[provider] = env_var_name

            # Dynamically check if the env var exists and create credential
            import os

            api_key = os.getenv(env_var_name)
            if api_key:
                credential = APIKeyCredentials(
                    id=f"{provider}-default",
                    provider=provider,
                    api_key=SecretStr(api_key),
                    title=f"Default {provider} credentials",
                )
                # Check if credential already exists to avoid duplicates
                if not any(c.id == credential.id for c in cls._default_credentials):
                    cls._default_credentials.append(credential)

    @classmethod
    def get_all_credentials(cls) -> List[Credentials]:
        """Replace hardcoded get_all_creds() in credentials_store.py."""
        with cls._lock:
            return cls._default_credentials.copy()

    @classmethod
    def get_oauth_handlers(cls) -> Dict[str, Type[BaseOAuthHandler]]:
        """Replace HANDLERS_BY_NAME in oauth/__init__.py."""
        with cls._lock:
            return cls._oauth_handlers.copy()

    @classmethod
    def get_oauth_credentials(cls) -> Dict[str, SDKOAuthCredentials]:
        """Get OAuth credentials configuration for SDK providers."""
        with cls._lock:
            return cls._oauth_credentials.copy()

    @classmethod
    def get_webhook_managers(cls) -> Dict[str, Type[BaseWebhooksManager]]:
        """Replace load_webhook_managers() in webhooks/__init__.py."""
        with cls._lock:
            return cls._webhook_managers.copy()

    @classmethod
    def register_block_configuration(
        cls, block_class: Type[Block], config: BlockConfiguration
    ) -> None:
        """Register configuration for a specific block class."""
        with cls._lock:
            cls._block_configurations[block_class] = config

    @classmethod
    def get_provider(cls, name: str) -> Optional["Provider"]:
        """Get a registered provider by name."""
        with cls._lock:
            return cls._providers.get(name)

    @classmethod
    def get_all_provider_names(cls) -> List[str]:
        """Get all registered provider names."""
        with cls._lock:
            return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        with cls._lock:
            cls._providers.clear()
            cls._default_credentials.clear()
            cls._oauth_handlers.clear()
            cls._webhook_managers.clear()
            cls._block_configurations.clear()
            cls._api_key_mappings.clear()

    @classmethod
    def patch_integrations(cls) -> None:
        """Patch existing integration points to use AutoRegistry."""
        # OAuth handlers are handled by SDKAwareHandlersDict in oauth/__init__.py
        # No patching needed for OAuth handlers

        # Patch webhook managers
        try:
            import sys
            from typing import Any

            # Get the module from sys.modules to respect mocking
            if "backend.integrations.webhooks" in sys.modules:
                webhooks: Any = sys.modules["backend.integrations.webhooks"]
            else:
                import backend.integrations.webhooks

                webhooks: Any = backend.integrations.webhooks

            if hasattr(webhooks, "load_webhook_managers"):
                original_load = webhooks.load_webhook_managers

                def patched_load():
                    # Get original managers
                    managers = original_load()
                    # Add SDK-registered managers
                    sdk_managers = cls.get_webhook_managers()
                    if isinstance(sdk_managers, dict):
                        # Import ProviderName for conversion
                        from backend.integrations.providers import ProviderName

                        # Convert string keys to ProviderName for consistency
                        for provider_str, manager in sdk_managers.items():
                            provider_name = ProviderName(provider_str)
                            managers[provider_name] = manager
                    return managers

                webhooks.load_webhook_managers = patched_load
        except Exception as e:
            logging.warning(f"Failed to patch webhook managers: {e}")
