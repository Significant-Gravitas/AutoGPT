from urllib.parse import urlparse

import fastapi

from backend.api.features.integrations.router import router as integrations_router
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks import utils as webhooks_utils


def test_webhook_ingress_url_matches_route(monkeypatch) -> None:
    app = fastapi.FastAPI()
    app.include_router(integrations_router, prefix="/api/integrations")

    provider = ProviderName.GITHUB
    webhook_id = "webhook_123"
    base_url = "https://example.com"

    monkeypatch.setattr(webhooks_utils.app_config, "platform_base_url", base_url)

    expected_path = app.url_path_for(
        "webhook_ingress_generic",
        provider=provider.value,
        webhook_id=webhook_id,
    )
    actual_path = urlparse(
        webhooks_utils.webhook_ingress_url(provider, webhook_id)
    ).path

    assert actual_path == expected_path
