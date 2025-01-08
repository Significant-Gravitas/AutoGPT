from backend.integrations.providers import ProviderName
from backend.util.settings import Config

app_config = Config()


# TODO: add test to assert this matches the actual API route
def webhook_ingress_url(provider_name: ProviderName, webhook_id: str) -> str:
    return (
        f"{app_config.platform_base_url}/api/integrations/{provider_name.value}"
        f"/webhooks/{webhook_id}/ingress"
    )
