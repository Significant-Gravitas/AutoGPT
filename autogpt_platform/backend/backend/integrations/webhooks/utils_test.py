import asyncio
import contextlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
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


@contextlib.asynccontextmanager
async def _null_transaction():
    """`transaction()` yields a client; the mocked models ignore it."""
    yield MagicMock()


class _Row:
    def __init__(self, id: str, name: str, data):
        self.id, self.name, self.data = id, name, data


def _preset(*, webhook_id=None, rows=()):
    preset = MagicMock()
    preset.id = "preset-1"
    preset.userId = "user-1"
    preset.agentGraphId = "graph-1"
    preset.agentGraphVersion = 1
    preset.webhookId = webhook_id
    preset.InputPresets = list(rows)
    return preset


def _graph(*, trigger_fields=("repo", "events"), graph_fields=("topic",)):
    graph = MagicMock()
    graph.webhook_input_node = MagicMock(id="trigger-node-1")
    graph.trigger_setup_info = MagicMock(
        config_schema={"properties": {name: {} for name in trigger_fields}}
    )
    graph.input_schema = {"properties": {name: {} for name in graph_fields}}
    return graph


def test_attached_preset_is_a_flat_trigger_config():
    """A preset holding a webhook is triggered by definition."""
    assert webhooks_utils._holds_flat_trigger_config(
        _preset(webhook_id="wh-1"), [], _graph()
    )


def test_detached_preset_with_trigger_only_field_is_recognised():
    """`remove_all_webhooks_for_credentials` nulls webhookId on intact triggered
    presets; a field only the trigger block declares still identifies them."""
    rows = [_Row("r1", "repo", "owner/repo")]
    assert webhooks_utils._holds_flat_trigger_config(_preset(rows=rows), rows, _graph())


def test_run_template_preset_is_left_alone():
    """A preset with real graph inputs and no webhook can live on a graph that
    merely contains a trigger node; folding its inputs would corrupt it."""
    rows = [_Row("r1", "topic", "weather")]
    assert not webhooks_utils._holds_flat_trigger_config(
        _preset(rows=rows), rows, _graph()
    )


def test_ambiguous_field_names_are_left_alone():
    """A name both schemas declare proves nothing, so the preset is skipped
    rather than guessed at."""
    rows = [_Row("r1", "repo", "owner/repo")]
    graph = _graph(trigger_fields=("repo",), graph_fields=("repo",))
    assert not webhooks_utils._holds_flat_trigger_config(
        _preset(rows=rows), rows, graph
    )


def test_graph_without_trigger_info_is_left_alone():
    graph = _graph()
    graph.trigger_setup_info = None
    rows = [_Row("r1", "repo", "owner/repo")]
    assert not webhooks_utils._holds_flat_trigger_config(
        _preset(rows=rows), rows, graph
    )


@pytest.mark.asyncio
async def test_backfill_costs_one_query_on_a_converged_database():
    """It runs on every REST boot inside a 30s budget, so the steady state must
    be a single selective query and no per-preset work at all."""
    presets = MagicMock()
    presets.prisma.return_value.find_many = AsyncMock(return_value=[])
    get_graph = AsyncMock()
    # The real registry, not a stub: the query's trigger-field arm below is
    # derived from it, so a stub would make that assertion compare [] to [].
    with (
        patch("prisma.models.AgentPreset", presets),
        patch("backend.data.graph.get_graph", get_graph),
    ):
        await webhooks_utils.migrate_flat_triggered_preset_inputs()

    presets.prisma.return_value.find_many.assert_awaited_once()
    get_graph.assert_not_awaited()
    # What makes the steady state empty: already-wrapped presets are excluded
    # by the query rather than skipped inside the loop.
    where = presets.prisma.return_value.find_many.await_args.kwargs["where"]
    assert where["InputPresets"] == {
        "none": {"name": {"startswith": "_node_input_mask_"}}
    }
    # ...and so are the un-convertible ones: a run-template preset is refused by
    # `_holds_flat_trigger_config` forever, at the price of a `get_graph` a boot.
    trigger_fields = webhooks_utils._trigger_config_field_names()
    assert where["OR"] == [
        {"NOT": [{"webhookId": None}]},
        {"InputPresets": {"some": {"name": {"in": trigger_fields}}}},
    ]
    # The arm that excludes it: trigger block field names are in, a graph input's
    # name is not.
    assert "repo" in trigger_fields and "events" in trigger_fields
    assert "topic" not in trigger_fields


@pytest.mark.asyncio
async def test_backfill_wraps_a_preset_the_sql_migration_missed():
    rows = [_Row("r1", "repo", "owner/repo"), _Row("r2", "events", ["push"])]
    preset = _preset(webhook_id="wh-1", rows=rows)

    presets = MagicMock()
    presets.prisma.return_value.find_many = AsyncMock(return_value=[preset])
    io_model = MagicMock()
    io_model.prisma.return_value.delete_many = AsyncMock(return_value=2)
    io_model.prisma.return_value.create = AsyncMock()

    with (
        patch("prisma.models.AgentPreset", presets),
        patch("prisma.models.AgentNodeExecutionInputOutput", io_model),
        patch("backend.data.graph.get_graph", AsyncMock(return_value=_graph())),
        patch("backend.blocks.get_webhook_block_ids", return_value=["block-1"]),
        patch("backend.data.db.transaction", _null_transaction),
    ):
        await webhooks_utils.migrate_flat_triggered_preset_inputs()

    # The flat rows are replaced by one mask row keyed on the trigger node, so
    # `_execute_webhook_preset_trigger` stops forwarding the config as graph input.
    created = io_model.prisma.return_value.create.await_args.kwargs["data"]
    assert created["name"] == "_node_input_mask_trigger"
    assert created["data"].data == {"repo": "owner/repo", "events": ["push"]}
    assert io_model.prisma.return_value.delete_many.await_args.kwargs["where"] == {
        "id": {"in": ["r1", "r2"]}
    }


@pytest.mark.asyncio
async def test_backfill_keeps_a_committed_page_when_the_budget_runs_out():
    """Boot bounds the backfill with `wait_for`, so the scan must be paged: a
    fetch that outlives the budget before any row commits never makes progress."""
    first = _preset(webhook_id="wh-1", rows=[_Row("r1", "repo", "owner/repo")])
    first.id = "preset-a"

    async def find_many(**kwargs):
        # An unbounded fetch, or any page after the first, outlives the budget.
        if "take" not in kwargs or "id" in kwargs["where"]:
            await asyncio.Event().wait()
        return [first]

    presets = MagicMock()
    presets.prisma.return_value.find_many = find_many
    io_model = MagicMock()
    io_model.prisma.return_value.delete_many = AsyncMock(return_value=1)
    io_model.prisma.return_value.create = AsyncMock()

    with (
        patch("prisma.models.AgentPreset", presets),
        patch("prisma.models.AgentNodeExecutionInputOutput", io_model),
        patch("backend.data.graph.get_graph", AsyncMock(return_value=_graph())),
        patch("backend.blocks.get_webhook_block_ids", return_value=["block-1"]),
        patch("backend.data.db.transaction", _null_transaction),
        patch.object(webhooks_utils, "_BACKFILL_PAGE_SIZE", 1),
        pytest.raises(asyncio.TimeoutError),
    ):
        await asyncio.wait_for(
            webhooks_utils.migrate_flat_triggered_preset_inputs(), timeout=1
        )

    io_model.prisma.return_value.create.assert_awaited_once()
    assert (
        io_model.prisma.return_value.create.await_args.kwargs["data"]["agentPresetId"]
        == "preset-a"
    )


@pytest.mark.asyncio
async def test_concurrent_backfills_convert_a_preset_once():
    """Every REST replica runs the backfill at boot. Against real Postgres, two
    that both read the same flat preset must leave one mask row, not two."""
    from prisma.models import (
        AgentGraph,
        AgentNode,
        AgentNodeExecutionInputOutput,
        AgentPreset,
    )

    from backend.api.features.library.model import node_input_mask_key
    from backend.blocks import get_webhook_block_ids
    from backend.data.user import get_or_create_user
    from backend.util.json import SafeJson

    user_id = str(uuid.uuid4())
    await get_or_create_user(
        {"sub": user_id, "email": f"backfill-{user_id}@example.com"}
    )
    graph = await AgentGraph.prisma().create(
        data={"id": str(uuid.uuid4()), "version": 1, "userId": user_id}
    )
    await AgentNode.prisma().create(
        data={
            "agentBlockId": sorted(get_webhook_block_ids())[0],
            "agentGraphId": graph.id,
            "agentGraphVersion": 1,
        }
    )
    preset = await AgentPreset.prisma().create(
        data={
            "name": "p",
            "description": "",
            "userId": user_id,
            "agentGraphId": graph.id,
            "agentGraphVersion": 1,
        }
    )
    for name, data in (("repo", "owner/repo"), ("events", ["push"])):
        await AgentNodeExecutionInputOutput.prisma().create(
            data={"name": name, "data": SafeJson(data), "agentPresetId": preset.id}
        )

    both_read = asyncio.Barrier(2)

    async def get_graph(graph_id, **_):
        # Other presets on a shared database are left alone.
        if graph_id != graph.id:
            return None
        await both_read.wait()
        return _graph()

    try:
        with patch("backend.data.graph.get_graph", get_graph):
            await asyncio.wait_for(
                asyncio.gather(
                    webhooks_utils.migrate_flat_triggered_preset_inputs(),
                    webhooks_utils.migrate_flat_triggered_preset_inputs(),
                ),
                timeout=60,
            )
        rows = await AgentNodeExecutionInputOutput.prisma().find_many(
            where={"agentPresetId": preset.id}
        )
        assert len(rows) == 1, [row.name for row in rows]
        assert rows[0].name == node_input_mask_key("trigger-node-1")
        assert rows[0].data == {"repo": "owner/repo", "events": ["push"]}
    finally:
        await AgentNodeExecutionInputOutput.prisma().delete_many(
            where={"agentPresetId": preset.id}
        )
        await AgentPreset.prisma().delete(where={"id": preset.id})
        await AgentGraph.prisma().delete_many(where={"id": graph.id})


def test_trigger_field_prefilter_covers_every_trigger_block():
    """Soundness of that SQL arm, against the real producer: any name the
    per-graph check can match lives in `trigger_setup_info.config_schema`, so
    every block's config schema must be inside the prefilter or the backfill
    would skip a preset it should convert."""
    import datetime

    from backend.blocks import get_webhook_block_ids
    from backend.data.graph import GraphModel, NodeModel

    prefilter = set(webhooks_utils._trigger_config_field_names())
    assert prefilter

    block_ids = list(get_webhook_block_ids())
    assert block_ids, "no trigger blocks loaded; the assertion below is vacuous"
    for block_id in block_ids:
        graph = GraphModel(
            id="graph-1",
            version=1,
            name="n",
            description="d",
            user_id="user-1",
            created_at=datetime.datetime.now(datetime.timezone.utc),
            nodes=[
                NodeModel(
                    id="11111111-2222-3333-4444-555555555555",
                    block_id=block_id,
                    graph_id="graph-1",
                    graph_version=1,
                )
            ],
            links=[],
        )
        trigger_info = graph.trigger_setup_info
        assert trigger_info, f"block #{block_id} has no trigger setup info"
        assert set(trigger_info.config_schema.get("properties", {})) <= prefilter
