from typing import TYPE_CHECKING

from .github import GithubWebhooksManager

if TYPE_CHECKING:
    from .base import BaseWebhooksManager

# --8<-- [start:WEBHOOK_MANAGERS_BY_NAME]
WEBHOOK_MANAGERS_BY_NAME: dict[str, type["BaseWebhooksManager"]] = {
    handler.PROVIDER_NAME: handler
    for handler in [
        GithubWebhooksManager,
    ]
}
# --8<-- [end:WEBHOOK_MANAGERS_BY_NAME]

__all__ = ["WEBHOOK_MANAGERS_BY_NAME"]
