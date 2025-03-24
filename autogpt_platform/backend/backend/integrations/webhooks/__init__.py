from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers import ProviderName
    from ._base import BaseWebhooksManager

_WEBHOOK_MANAGERS: dict["ProviderName", type["BaseWebhooksManager"]] = {}


# --8<-- [start:load_webhook_managers]
def load_webhook_managers() -> dict["ProviderName", type["BaseWebhooksManager"]]:
    if _WEBHOOK_MANAGERS:
        return _WEBHOOK_MANAGERS

    from .compass import CompassWebhookManager
    from .github import GithubWebhooksManager
    from .slant3d import Slant3DWebhooksManager

    _WEBHOOK_MANAGERS.update(
        {
            handler.PROVIDER_NAME: handler
            for handler in [
                CompassWebhookManager,
                GithubWebhooksManager,
                Slant3DWebhooksManager,
            ]
        }
    )
    return _WEBHOOK_MANAGERS


# --8<-- [end:load_webhook_managers]


def get_webhook_manager(provider_name: "ProviderName") -> "BaseWebhooksManager":
    return load_webhook_managers()[provider_name]()


def supports_webhooks(provider_name: "ProviderName") -> bool:
    return provider_name in load_webhook_managers()


__all__ = ["get_webhook_manager", "supports_webhooks"]
