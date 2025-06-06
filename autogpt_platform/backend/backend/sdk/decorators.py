"""
Registration Decorators for AutoGPT Platform Blocks

These decorators allow blocks to self-register their configurations:
- @cost_config: Register block cost configuration
- @default_credentials: Register default platform credentials
- @provider: Register new provider name
- @webhook_config: Register webhook manager
- @oauth_config: Register OAuth handler
"""

from typing import Any, List, Optional, Type

from .auto_registry import get_registry


def cost_config(*cost_configurations):
    """
    Decorator to register cost configuration for a block.

    Usage:
        @cost_config(
            BlockCost(cost_amount=5, cost_type=BlockCostType.RUN),
            BlockCost(cost_amount=1, cost_type=BlockCostType.BYTE)
        )
        class MyBlock(Block):
            pass
    """

    def decorator(block_class: Type):
        registry = get_registry()
        registry.register_block_cost(block_class, list(cost_configurations))
        return block_class

    return decorator


def default_credentials(*credentials):
    """
    Decorator to register default platform credentials.

    Usage:
        @default_credentials(
            APIKeyCredentials(
                id="myservice-default",
                provider="myservice",
                api_key=SecretStr("default-key"),
                title="MyService Default API Key"
            )
        )
        class MyBlock(Block):
            pass
    """

    def decorator(block_class: Type):
        registry = get_registry()
        for credential in credentials:
            registry.register_default_credential(credential)
        return block_class

    return decorator


def provider(provider_name: str):
    """
    Decorator to register a new provider name.

    Usage:
        @provider("myservice")
        class MyBlock(Block):
            pass
    """

    def decorator(block_class: Type):
        registry = get_registry()
        registry.register_provider(provider_name)
        # Also ensure the provider is registered in the block class
        if hasattr(block_class, "_provider"):
            block_class._provider = provider_name
        return block_class

    return decorator


def webhook_config(provider_name: str, manager_class: Type):
    """
    Decorator to register a webhook manager.

    Usage:
        @webhook_config("github", GitHubWebhooksManager)
        class GitHubWebhookBlock(Block):
            pass
    """

    def decorator(block_class: Type):
        registry = get_registry()
        registry.register_webhook_manager(provider_name, manager_class)
        # Store webhook manager reference on block class
        if hasattr(block_class, "_webhook_manager"):
            block_class._webhook_manager = manager_class
        return block_class

    return decorator


def oauth_config(provider_name: str, handler_class: Type):
    """
    Decorator to register an OAuth handler.

    Usage:
        @oauth_config("github", GitHubOAuthHandler)
        class GitHubBlock(Block):
            pass
    """

    def decorator(block_class: Type):
        registry = get_registry()
        registry.register_oauth_handler(provider_name, handler_class)
        # Store OAuth handler reference on block class
        if hasattr(block_class, "_oauth_handler"):
            block_class._oauth_handler = handler_class
        return block_class

    return decorator


# === CONVENIENCE DECORATORS ===
def register_credentials(*credentials):
    """Alias for default_credentials decorator."""
    return default_credentials(*credentials)


def register_cost(*cost_configurations):
    """Alias for cost_config decorator."""
    return cost_config(*cost_configurations)


def register_oauth(provider_name: str, handler_class: Type):
    """Alias for oauth_config decorator."""
    return oauth_config(provider_name, handler_class)


def register_webhook_manager(provider_name: str, manager_class: Type):
    """Alias for webhook_config decorator."""
    return webhook_config(provider_name, manager_class)


# === COMBINATION DECORATOR ===
def block_config(
    provider_name: Optional[str] = None,
    costs: Optional[List[Any]] = None,
    credentials: Optional[List[Any]] = None,
    oauth_handler: Optional[Type] = None,
    webhook_manager: Optional[Type] = None,
):
    """
    Combined decorator for all block configurations.

    Usage:
        @block_config(
            provider_name="myservice",
            costs=[BlockCost(cost_amount=5, cost_type=BlockCostType.RUN)],
            credentials=[APIKeyCredentials(...)],
            oauth_handler=MyServiceOAuthHandler,
            webhook_manager=MyServiceWebhookManager
        )
        class MyServiceBlock(Block):
            pass
    """

    def decorator(block_class: Type):
        registry = get_registry()

        if provider_name:
            registry.register_provider(provider_name)
            if hasattr(block_class, "_provider"):
                block_class._provider = provider_name

        if costs:
            registry.register_block_cost(block_class, costs)

        if credentials:
            for credential in credentials:
                registry.register_default_credential(credential)

        if oauth_handler and provider_name:
            registry.register_oauth_handler(provider_name, oauth_handler)
            if hasattr(block_class, "_oauth_handler"):
                block_class._oauth_handler = oauth_handler

        if webhook_manager and provider_name:
            registry.register_webhook_manager(provider_name, webhook_manager)
            if hasattr(block_class, "_webhook_manager"):
                block_class._webhook_manager = webhook_manager

        return block_class

    return decorator
