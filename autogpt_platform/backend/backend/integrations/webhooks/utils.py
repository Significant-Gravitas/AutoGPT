from backend.util.settings import Config

app_config = Config()


# TODO: add test to assert this matches the actual API route
def webhook_ingress_url(provider_name: str, webhook_id: str) -> str:
    return (
        f"{app_config.platform_base_url}/api/integrations/{provider_name}"
        f"/webhooks/{webhook_id}/ingress"
    )
