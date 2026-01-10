from urllib.parse import urlparse

import fastapi
from fastapi.routing import APIRoute

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

    route = next(
        route
        for route in integrations_router.routes
        if isinstance(route, APIRoute)
        and route.path == "/{provider}/webhooks/{webhook_id}/ingress"
        and "POST" in route.methods
    )
    expected_path = f"/api/integrations{route.path}".format(
        provider=provider.value,
        webhook_id=webhook_id,
    )
    actual_url = urlparse(webhooks_utils.webhook_ingress_url(provider, webhook_id))
    expected_base = urlparse(base_url)

    assert (actual_url.scheme, actual_url.netloc) == (
        expected_base.scheme,
        expected_base.netloc,
    )
    assert actual_url.path == expected_path
