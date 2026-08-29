from unittest.mock import AsyncMock, patch
from urllib.parse import urlparse

import fastapi
import pytest
from fastapi.routing import APIRoute

from backend.api.features.integrations.router import router as integrations_router
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks import utils as webhooks_utils
from backend.integrations.webhooks.github import GithubWebhooksManager


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


@pytest.mark.asyncio
async def test_get_manual_webhook_tags_webhook_with_parent_tenant(
    monkeypatch,
) -> None:
    """A manual webhook created for a graph/preset must carry the parent
    resource's org/team (resource-follows-parent) down to the DB row."""
    from backend.data import integrations
    from backend.integrations.webhooks import _base as webhooks_base

    # _base has its own module-level Config() — get_manual_webhook checks
    # that one, not utils.app_config, so patch both.
    monkeypatch.setattr(
        webhooks_utils.app_config, "platform_base_url", "https://example.com"
    )
    monkeypatch.setattr(
        webhooks_base.app_config, "platform_base_url", "https://example.com"
    )
    manager = GithubWebhooksManager()

    captured: dict = {}

    async def fake_create(webhook: integrations.Webhook) -> integrations.Webhook:
        captured["webhook"] = webhook
        return webhook

    with (
        patch.object(
            integrations,
            "find_webhook_by_graph_and_props",
            AsyncMock(return_value=None),
        ),
        patch.object(integrations, "create_webhook", side_effect=fake_create),
    ):
        webhook = await manager.get_manual_webhook(
            user_id="u1",
            webhook_type=GithubWebhooksManager.WebhookType.REPO,
            events=["push"],
            graph_id="g-1",
            organization_id="org-parent",
            team_id="team-parent",
        )

    assert webhook.organization_id == "org-parent"
    assert webhook.team_id == "team-parent"
    assert captured["webhook"].organization_id == "org-parent"
    assert captured["webhook"].team_id == "team-parent"
