"""Saving a change to a webhook trigger's settings, through the presets route
and a real database (#17382).

The Library sends a trigger's inputs and keys together, which re-registers its
webhook and then attaches it to the preset. For an expert's trigger, attaching
it also sets the preset's organization and team. Setting the team as a column
beside the webhook relation is an update Prisma refuses, so the route answered
500 after the new settings were already stored. A mocked client accepts any
update, which is why this needs the database.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import fastapi
import httpx
import prisma.models
import pytest
import pytest_asyncio
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.experts import experts_db
from backend.api.features.library.routes.presets import router
from backend.api.features.orgs.db import get_user_default_team
from backend.api.model import CreateGraph
from backend.api.rest_api import app as rest_app
from backend.blocks import _base as block_base
from backend.blocks.generic_webhook.triggers import GenericWebhookTriggerBlock
from backend.data.graph import Graph, Node
from backend.data.user import get_or_create_user
from backend.integrations.webhooks import _base as webhooks_base
from backend.util.test import SpinTestServer

PRESETS = "/api/library/presets"
OLD = {"source": "sales"}
NEW = {"source": "billing"}

app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
# Mounted where rest_api.py mounts it.
app.include_router(router, prefix="/api/library")


@pytest.fixture(autouse=True)
def webhooks_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # Trigger blocks and webhooks need a platform URL, which CI doesn't set.
    for config in (block_base.app_config, webhooks_base.app_config):
        monkeypatch.setattr(config, "platform_base_url", "https://example.com")


@pytest_asyncio.fixture(loop_scope="session")
async def owner(server: SpinTestServer) -> AsyncIterator[str]:
    """A user of their own, deleted afterwards with what the test made."""
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"trigger-save-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Trigger Owner",
        }
    )
    try:
        yield user.id
    finally:
        # A webhook keeps its user from being deleted. The personal org has no
        # FK to its user; the user takes its graphs, presets and library agents.
        await prisma.models.IntegrationWebhook.prisma().delete_many(
            where={"userId": user.id}
        )
        await prisma.models.Expert.prisma().delete_many(where={"ownerUserId": user.id})
        await prisma.models.Organization.prisma().delete_many(
            where={"bootstrapUserId": user.id}
        )
        await prisma.models.User.prisma().delete_many(where={"id": user.id})


@pytest_asyncio.fixture(loop_scope="session")
async def api(owner: str) -> AsyncIterator[httpx.AsyncClient]:
    """Signed in as the owner; only the token check is overridden."""
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": owner,
        "role": "user",
        "email": "owner@example.com",
    }
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_jwt_payload, None)


@pytest.mark.asyncio(loop_scope="session")
async def test_an_experts_trigger_saves_its_new_settings(
    server: SpinTestServer, owner: str, api: httpx.AsyncClient
):
    expert = await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": owner,
            "name": "Scout",
            "role": "Operations",
            "identity": "You are Scout, an expert.",
        }
    )
    trigger = await _set_up_trigger(server, api, owner, expert_id=expert.id)
    assert trigger["expert_id"] == expert.id and trigger["webhook_id"]
    [mask] = trigger["inputs"]

    saved = await _change_settings(api, trigger)

    assert saved.status_code == 200, saved.text
    stored = await prisma.models.AgentPreset.prisma().find_unique_or_raise(
        where={"id": trigger["id"]}, include={"InputPresets": True}
    )
    assert {row.name: row.data for row in stored.InputPresets or []} == {
        mask: {"constants": NEW}
    }
    # A manual webhook keeps its URL, and an expert's work stays in its
    # owner's personal team.
    assert (stored.webhookId, stored.expertId) == (trigger["webhook_id"], expert.id)
    assert (stored.organizationId, stored.teamId) == await get_user_default_team(owner)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_trigger_of_no_expert_saves_its_new_settings(
    server: SpinTestServer, owner: str, api: httpx.AsyncClient
):
    """The control: the same change, to a trigger that belongs to no expert."""
    trigger = await _set_up_trigger(server, api, owner)
    assert trigger["expert_id"] is None and trigger["webhook_id"]

    saved = await _change_settings(api, trigger)

    assert saved.status_code == 200, saved.text


async def _set_up_trigger(
    server: SpinTestServer,
    api: httpx.AsyncClient,
    owner: str,
    *,
    expert_id: str | None = None,
) -> dict[str, Any]:
    """A workflow a manual webhook starts, installed on the expert when there
    is one, and its trigger set up through the route."""
    graph = await server.agent_server.test_create_graph(
        CreateGraph(
            graph=Graph(
                name="Inbound webhook",
                description="",
                nodes=[Node(block_id=GenericWebhookTriggerBlock().id)],
                links=[],
            )
        ),
        owner,
    )
    if expert_id:
        library_agent = await prisma.models.LibraryAgent.prisma().find_first_or_raise(
            where={"userId": owner, "agentGraphId": graph.id}
        )
        await experts_db.install_workflow(
            owner, expert_id, library_agent_id=library_agent.id
        )
    set_up = await api.post(
        f"{PRESETS}/setup-trigger",
        json={
            "name": "Inbound webhook",
            "graph_id": graph.id,
            "graph_version": graph.version,
            "trigger_config": {"constants": OLD},
        },
    )
    assert set_up.status_code == 200, set_up.text
    return set_up.json()


async def _change_settings(
    api: httpx.AsyncClient, trigger: dict[str, Any]
) -> httpx.Response:
    """What the Library's trigger view sends on Save: the inputs and the keys
    together (none, for a workflow that needs none)."""
    [(mask, config)] = trigger["inputs"].items()
    return await api.patch(
        f"{PRESETS}/{trigger['id']}",
        json={"inputs": {mask: {**config, "constants": NEW}}, "credentials": {}},
    )
