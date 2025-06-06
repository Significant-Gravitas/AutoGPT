"""
Auto-Registration System for AutoGPT Platform

Automatically discovers and registers:
- Block costs
- Default credentials  
- OAuth handlers
- Webhook managers
- Provider names

This eliminates the need to manually update configuration files
outside the blocks folder when adding new blocks.
"""

from typing import Any, Dict, List, Set, Type


# === GLOBAL REGISTRIES ===
class AutoRegistry:
    """Central registry for auto-discovered block configurations."""

    def __init__(self):
        self.block_costs: Dict[Type, List] = {}
        self.default_credentials: List[Any] = []
        self.oauth_handlers: Dict[str, Type] = {}
        self.webhook_managers: Dict[str, Type] = {}
        self.providers: Set[str] = set()

    def register_block_cost(self, block_class: Type, cost_config: List):
        """Register cost configuration for a block."""
        self.block_costs[block_class] = cost_config

    def register_default_credential(self, credential):
        """Register a default platform credential."""
        # Avoid duplicates based on provider and id
        for existing in self.default_credentials:
            if (
                hasattr(existing, "provider")
                and hasattr(credential, "provider")
                and existing.provider == credential.provider
                and hasattr(existing, "id")
                and hasattr(credential, "id")
                and existing.id == credential.id
            ):
                return  # Skip duplicate
        self.default_credentials.append(credential)

    def register_oauth_handler(self, provider_name: str, handler_class: Type):
        """Register an OAuth handler for a provider."""
        self.oauth_handlers[provider_name] = handler_class

    def register_webhook_manager(self, provider_name: str, manager_class: Type):
        """Register a webhook manager for a provider."""
        self.webhook_managers[provider_name] = manager_class

    def register_provider(self, provider_name: str):
        """Register a new provider name."""
        self.providers.add(provider_name)

    def get_block_costs_dict(self) -> Dict[Type, List]:
        """Get block costs in format expected by current system."""
        return self.block_costs.copy()

    def get_default_credentials_list(self) -> List[Any]:
        """Get default credentials in format expected by current system."""
        return self.default_credentials.copy()

    def get_oauth_handlers_dict(self) -> Dict[str, Type]:
        """Get OAuth handlers in format expected by current system."""
        return self.oauth_handlers.copy()

    def get_webhook_managers_dict(self) -> Dict[str, Type]:
        """Get webhook managers in format expected by current system."""
        return self.webhook_managers.copy()


# Global registry instance
_registry = AutoRegistry()


def get_registry() -> AutoRegistry:
    """Get the global auto-registry instance."""
    return _registry


# === DISCOVERY FUNCTIONS ===
def discover_block_configurations():
    """
    Discover all block configurations by scanning loaded blocks.
    Called during application startup after blocks are loaded.
    """
    from backend.blocks import load_all_blocks

    # Load all blocks (this also imports all block modules)
    load_all_blocks()  # This triggers decorator execution

    # Registry is populated by decorators during import
    return _registry


def patch_existing_systems():
    """
    Patch existing configuration systems to use auto-discovered data.
    This maintains backward compatibility while enabling auto-registration.
    """

    # Patch block cost configuration
    try:
        import backend.data.block_cost_config as cost_config

        original_block_costs = getattr(cost_config, "BLOCK_COSTS", {})
        # Merge auto-registered costs with existing ones
        merged_costs = {**original_block_costs}
        merged_costs.update(_registry.get_block_costs_dict())
        cost_config.BLOCK_COSTS = merged_costs
    except Exception as e:
        print(f"Warning: Could not patch block cost config: {e}")

    # Patch credentials store
    try:
        import backend.integrations.credentials_store as cred_store

        if hasattr(cred_store, "DEFAULT_CREDENTIALS"):
            # Add auto-registered credentials to the existing list
            for cred in _registry.get_default_credentials_list():
                if cred not in cred_store.DEFAULT_CREDENTIALS:
                    cred_store.DEFAULT_CREDENTIALS.append(cred)

        # Also patch the IntegrationCredentialsStore.get_all_creds method
        if hasattr(cred_store, "IntegrationCredentialsStore"):
            original_get_all_creds = (
                cred_store.IntegrationCredentialsStore.get_all_creds
            )

            def patched_get_all_creds(self, user_id: str) -> list:
                # Get original credentials list
                creds_list = original_get_all_creds(self, user_id)

                # Add auto-registered credentials that aren't already in the list
                for credential in _registry.get_default_credentials_list():
                    # Check if credential is already in list by id
                    if not any(
                        hasattr(c, "id") and c.id == credential.id for c in creds_list
                    ):
                        creds_list.append(credential)

                return creds_list

            cred_store.IntegrationCredentialsStore.get_all_creds = patched_get_all_creds
    except Exception as e:
        print(f"Warning: Could not patch credentials store: {e}")

    # Patch OAuth handlers
    try:
        import backend.integrations.oauth as oauth_module

        if hasattr(oauth_module, "HANDLERS_BY_NAME"):
            # Convert string keys to ProviderName enum
            from backend.integrations.providers import ProviderName

            for provider_str, handler in _registry.get_oauth_handlers_dict().items():
                provider_enum = ProviderName(provider_str)
                oauth_module.HANDLERS_BY_NAME[provider_enum] = handler
    except Exception as e:
        print(f"Warning: Could not patch OAuth handlers: {e}")

    # Patch webhook managers
    try:
        import backend.integrations.webhooks as webhook_module

        if hasattr(webhook_module, "_WEBHOOK_MANAGERS"):
            # Convert string keys to ProviderName enum
            from backend.integrations.providers import ProviderName

            for provider_str, manager in _registry.get_webhook_managers_dict().items():
                provider_enum = ProviderName(provider_str)
                webhook_module._WEBHOOK_MANAGERS[provider_enum] = manager

        # Also patch the load_webhook_managers function
        if hasattr(webhook_module, "load_webhook_managers"):
            original_load = webhook_module.load_webhook_managers

            def patched_load_webhook_managers():
                # Call original to load existing managers
                managers = original_load()
                # Add auto-registered managers
                from backend.integrations.providers import ProviderName

                for (
                    provider_str,
                    manager,
                ) in _registry.get_webhook_managers_dict().items():
                    provider_enum = ProviderName(provider_str)
                    managers[provider_enum] = manager
                return managers

            webhook_module.load_webhook_managers = patched_load_webhook_managers
    except Exception as e:
        print(f"Warning: Could not patch webhook managers: {e}")

    # Extend provider enum dynamically
    try:
        from backend.integrations.providers import ProviderName

        for provider_name in _registry.providers:
            # Add provider to enum if not already present
            if not any(member.value == provider_name for member in ProviderName):
                # This is tricky with enums, so we'll store for reference
                # In practice, we might need to handle this differently
                pass
    except Exception as e:
        print(f"Warning: Could not extend provider enum: {e}")


def setup_auto_registration():
    """
    Set up the auto-registration system.
    This should be called during application startup after blocks are loaded.
    """
    # Discover all block configurations
    registry = discover_block_configurations()

    # Patch existing systems to use discovered configurations
    patch_existing_systems()

    # Log registration results
    print("Auto-registration complete:")
    print(f"  - {len(registry.block_costs)} block costs registered")
    print(f"  - {len(registry.default_credentials)} default credentials registered")
    print(f"  - {len(registry.oauth_handlers)} OAuth handlers registered")
    print(f"  - {len(registry.webhook_managers)} webhook managers registered")
    print(f"  - {len(registry.providers)} providers registered")

    return registry
