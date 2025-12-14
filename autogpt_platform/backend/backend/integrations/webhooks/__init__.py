from typing import TYPE_CHECKING

from backend.util.cache import cached

if TYPE_CHECKING:
    from ..providers import ProviderName
    from ._base import BaseWebhooksManager


# --8<-- [start:load_webhook_managers]
@cached(ttl_seconds=3600)
def load_webhook_managers() -> dict["ProviderName", type["BaseWebhooksManager"]]:
    webhook_managers = {}

    from .compass import CompassWebhookManager
    from .github import GithubWebhooksManager
    from .slant3d import Slant3DWebhooksManager

    webhook_managers.update(
        {
            handler.PROVIDER_NAME: handler
            for handler in [
                CompassWebhookManager,
                GithubWebhooksManager,
                Slant3DWebhooksManager,
            ]
        }
    )
    return webhook_managers


# --8<-- [end:load_webhook_managers]


def get_webhook_manager(provider_name: "ProviderName") -> "BaseWebhooksManager":
    return load_webhook_managers()[provider_name]()


def supports_webhooks(provider_name: "ProviderName") -> bool:
    return provider_name in load_webhook_managers()


__all__ = ["get_webhook_manager", "supports_webhooks"]
