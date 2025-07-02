"""
Auto-registration system for blocks, providers, and their configurations.
"""

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from pydantic import SecretStr

from backend.blocks.basic import Block
from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.integrations.webhooks._base import BaseWebhooksManager

if TYPE_CHECKING:
    from backend.sdk.provider import Provider


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
    _webhook_managers: Dict[str, Type[BaseWebhooksManager]] = {}
    _block_configurations: Dict[Type[Block], BlockConfiguration] = {}
    _api_key_mappings: Dict[str, str] = {}  # provider -> env_var_name

    @classmethod
    def register_provider(cls, provider: "Provider") -> None:
        """Auto-register provider and all its configurations."""
        with cls._lock:
            cls._providers[provider.name] = provider

            # Register OAuth handler if provided
            if provider.oauth_handler:
                cls._oauth_handlers[provider.name] = provider.oauth_handler

            # Register webhook manager if provided
            if provider.webhook_manager:
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
        # Patch oauth handlers
        try:
            import backend.integrations.oauth as oauth

            if hasattr(oauth, "HANDLERS_BY_NAME"):
                # Create a new dict that includes both original and SDK handlers
                original_handlers = dict(oauth.HANDLERS_BY_NAME)

                class PatchedHandlersDict(dict):  # type: ignore
                    def __getitem__(self, key):
                        # First try SDK handlers
                        sdk_handlers = cls.get_oauth_handlers()
                        if key in sdk_handlers:
                            return sdk_handlers[key]
                        # Fall back to original
                        return original_handlers[key]

                    def get(self, key, default=None):
                        try:
                            return self[key]
                        except KeyError:
                            return default

                    def __contains__(self, key):
                        sdk_handlers = cls.get_oauth_handlers()
                        return key in sdk_handlers or key in original_handlers

                    def keys(self):  # type: ignore[override]
                        sdk_handlers = cls.get_oauth_handlers()
                        all_keys = set(original_handlers.keys()) | set(
                            sdk_handlers.keys()
                        )
                        return all_keys

                    def values(self):
                        combined = dict(original_handlers)
                        sdk_handlers = cls.get_oauth_handlers()
                        if isinstance(sdk_handlers, dict):
                            combined.update(sdk_handlers)  # type: ignore
                        return combined.values()

                    def items(self):
                        combined = dict(original_handlers)
                        sdk_handlers = cls.get_oauth_handlers()
                        if isinstance(sdk_handlers, dict):
                            combined.update(sdk_handlers)  # type: ignore
                        return combined.items()

                oauth.HANDLERS_BY_NAME = PatchedHandlersDict()
        except Exception as e:
            logging.warning(f"Failed to patch oauth handlers: {e}")

        # Patch webhook managers
        try:
            import backend.integrations.webhooks as webhooks

            if hasattr(webhooks, "load_webhook_managers"):
                original_load = webhooks.load_webhook_managers

                def patched_load():
                    # Get original managers
                    managers = original_load()
                    # Add SDK-registered managers
                    sdk_managers = cls.get_webhook_managers()
                    if isinstance(sdk_managers, dict):
                        managers.update(sdk_managers)  # type: ignore
                    return managers

                webhooks.load_webhook_managers = patched_load
        except Exception as e:
            logging.warning(f"Failed to patch webhook managers: {e}")
