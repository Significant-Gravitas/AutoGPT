from typing import TYPE_CHECKING, Optional

from backend.integrations.providers import ProviderName
from backend.util.settings import Config

from . import get_webhook_manager, supports_webhooks

if TYPE_CHECKING:
    from backend.data.integrations import Webhook
    from backend.data.model import Credentials

app_config = Config()


# TODO: add test to assert this matches the actual API route
def webhook_ingress_url(provider_name: ProviderName, webhook_id: str) -> str:
    return (
        f"{app_config.platform_base_url}/api/integrations/{provider_name.value}"
        f"/webhooks/{webhook_id}/ingress"
    )


async def setup_webhook(
    user_id: str,
    provider: ProviderName,
    webhook_type: str,
    credentials: Optional["Credentials"] = None,
    resource: Optional[str] = None,
    events: Optional[list[str]] = None,
    for_graph_id: Optional[str] = None,
    for_preset_id: Optional[str] = None,
) -> "Webhook":
    """
    Utility function to create (and auto-setup if possible) a webhook for a given provider.

    Either `for_graph_id` or `for_preset_id` must be provided if the webhook is
    being set up manually (i.e., not auto-setup).
    """
    from ._manual_base import ManualWebhookManagerBase

    if not supports_webhooks(provider):
        raise ValueError(f"Provider {provider} does not support webhooks")

    webhooks_manager = get_webhook_manager(provider)

    auto_setup_webhook = isinstance(webhooks_manager, ManualWebhookManagerBase)

    if auto_setup_webhook:
        if not resource:
            raise ValueError(
                f"Cannot auto-setup webhook for provider {provider} without resource"
            )

        if not credentials:
            raise ValueError(
                f"Cannot set up webhook for provider {provider} without credentials"
            )
        elif credentials.provider != provider:
            raise ValueError(
                f"Credentials #{credentials.id} do not match provider {provider}"
            )

    webhooks_manager = get_webhook_manager(provider)

    # Find/make and attach a suitable webhook to the node
    if auto_setup_webhook:
        assert resource is not None
        assert credentials is not None
        new_webhook = await webhooks_manager.get_suitable_auto_webhook(
            user_id,
            credentials,
            webhook_type,
            resource,
            events or [],
        )
    else:
        # Manual webhook -> no credentials -> don't register but do create
        new_webhook = await webhooks_manager.get_manual_webhook(
            user_id,
            webhook_type,
            events or [],
            graph_id=for_graph_id,
            preset_id=for_preset_id,
        )
    return new_webhook
