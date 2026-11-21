"""Reads of the triggers attached to a webhook; real database."""

import uuid

import prisma.models
import pytest

from backend.data import integrations
from backend.data.user import get_or_create_user
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer


@pytest.mark.parametrize("attached, pages", [(7, [3, 3, 1]), (6, [3, 3])])
@pytest.mark.asyncio(loop_scope="session")
async def test_a_webhooks_presets_come_a_page_at_a_time_with_their_inputs(
    server: SpinTestServer, attached: int, pages: list[int]
):
    user_id = await _user()
    graph_id = await _graph(user_id)
    webhook_id = await _webhook(user_id)
    preset_ids = {
        await _preset(user_id, graph_id, webhook_id, topic=f"t{n}")
        for n in range(attached)
    }
    await _preset(user_id, graph_id, await _webhook(user_id), topic="elsewhere")

    read = [
        page
        async for page in integrations.iter_webhook_triggered_presets(webhook_id, 3)
    ]

    assert [len(page) for page in read] == pages
    assert {preset.id for page in read for preset in page} == preset_ids
    assert {preset.inputs["topic"] for page in read for preset in page} == {
        f"t{n}" for n in range(attached)
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_a_webhooks_triggers_are_found_and_detached_by_graph(
    server: SpinTestServer,
):
    user_id = await _user()
    stale_graph, kept_graph = await _graph(user_id), await _graph(user_id)
    webhook_id, unused_webhook_id = await _webhook(user_id), await _webhook(user_id)
    for graph_id in (stale_graph, stale_graph, kept_graph):
        await _preset(user_id, graph_id, webhook_id, topic="t")

    assert await integrations.webhooks_have_triggers([unused_webhook_id, webhook_id])
    assert not await integrations.webhooks_have_triggers([unused_webhook_id])

    await integrations.unlink_webhook_triggers(
        webhook_id, user_id, graph_id=stale_graph
    )
    assert {p.agentGraphId for p in await _attached(webhook_id)} == {kept_graph}

    await integrations.unlink_webhook_triggers(webhook_id, user_id)
    assert not await integrations.webhooks_have_triggers([webhook_id])


async def _user() -> str:
    user_id = str(uuid.uuid4())
    await get_or_create_user({"sub": user_id, "email": f"wh-{user_id}@example.com"})
    return user_id


async def _graph(user_id: str) -> str:
    graph = await prisma.models.AgentGraph.prisma().create(
        data={"id": str(uuid.uuid4()), "version": 1, "userId": user_id}
    )
    return graph.id


async def _webhook(user_id: str) -> str:
    webhook = await prisma.models.IntegrationWebhook.prisma().create(
        data={
            "userId": user_id,
            "provider": "telegram",
            "credentialsId": "",
            "webhookType": "bot",
            "resource": "",
            "events": [],
            "config": "{}",
            "secret": "",
            "providerWebhookId": "",
        }
    )
    return webhook.id


async def _preset(user_id: str, graph_id: str, webhook_id: str, topic: str) -> str:
    preset = await prisma.models.AgentPreset.prisma().create(
        data={
            "name": "p",
            "description": "",
            "userId": user_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "webhookId": webhook_id,
            "InputPresets": {
                "create": [{"name": "topic", "data": SafeJson(topic)}],
            },
        }
    )
    return preset.id


async def _attached(webhook_id: str) -> list[prisma.models.AgentPreset]:
    return await prisma.models.AgentPreset.prisma().find_many(
        where={"webhookId": webhook_id}
    )
