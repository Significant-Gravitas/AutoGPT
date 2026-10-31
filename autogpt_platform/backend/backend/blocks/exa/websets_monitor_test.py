"""Payload tests for the Websets monitor, monitor-run, team and import blocks."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.exa.websets_fixtures import (
    IMPORT_ID,
    MONITOR_ID,
    WEBSET_ID,
    collect_outputs,
    sample_monitor,
    sample_monitor_run,
    sample_team,
)
from backend.blocks.exa.websets_import_update import ExaUpdateImportBlock
from backend.blocks.exa.websets_monitor import (
    ExaCreateMonitorBlock,
    ExaGetMonitorBlock,
    ExaUpdateMonitorBlock,
)
from backend.blocks.exa.websets_monitor_runs import ExaGetWebsetMonitorRunBlock
from backend.blocks.exa.websets_team import ExaGetWebsetsTeamInfoBlock


@pytest.mark.asyncio
async def test_create_monitor_omits_unset_query_so_exa_reuses_last_search():
    block = ExaCreateMonitorBlock()
    api = AsyncMock(return_value=sample_monitor())
    with patch.object(block, "_api", api):
        await collect_outputs(
            block,
            webset_id=WEBSET_ID,
            cron_expression="0 9 * * *",
            timezone="Europe/London",
            entity_type="custom",
            entity_description="Seed-stage startups",
        )

    assert api.call_args.args[1:3] == ("POST", "/monitors")
    assert api.call_args.kwargs["json"] == {
        "websetId": WEBSET_ID,
        "cadence": {"cron": "0 9 * * *", "timezone": "Europe/London"},
        "behavior": {
            "type": "search",
            "config": {
                "count": 10,
                "entity": {"type": "custom", "description": "Seed-stage startups"},
                "behavior": "append",
            },
        },
    }


@pytest.mark.asyncio
async def test_create_monitor_rejects_legacy_refresh():
    block = ExaCreateMonitorBlock()
    api = AsyncMock()
    with patch.object(block, "_api", api), pytest.raises(
        ValueError, match="refresh monitors"
    ):
        await collect_outputs(
            block,
            webset_id=WEBSET_ID,
            cron_expression="0 9 * * *",
            behavior_type="refresh",
        )
    api.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_monitor_tolerates_failed_last_run():
    block = ExaGetMonitorBlock()
    monitor = sample_monitor(lastRun=sample_monitor_run(status="failed"))
    with patch.object(block, "_api", AsyncMock(return_value=monitor)):
        outputs = dict(await collect_outputs(block, monitor_id=MONITOR_ID))

    assert outputs["last_run"]["status"] == "failed"
    assert outputs["cron_expression"] == "0 9 * * 1"


@pytest.mark.asyncio
async def test_update_monitor_maps_paused_and_sends_cadence_and_search():
    block = ExaUpdateMonitorBlock()
    api = AsyncMock(return_value=sample_monitor(status="disabled"))
    with patch.object(block, "_api", api):
        await collect_outputs(
            block,
            monitor_id=MONITOR_ID,
            status="paused",
            cron_expression="0 8 * * 1",
            timezone="America/New_York",
            search_query="New AI startups",
            search_count=5,
            search_behavior="override",
        )

    assert api.call_args.args[1:3] == ("PATCH", f"/monitors/{MONITOR_ID}")
    assert api.call_args.kwargs["json"] == {
        "status": "disabled",
        "cadence": {"cron": "0 8 * * 1", "timezone": "America/New_York"},
        "behavior": {
            "type": "search",
            "config": {"count": 5, "query": "New AI startups", "behavior": "override"},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inputs, message",
    [
        ({"timezone": "Europe/Paris"}, "cron expression"),
        ({"search_query": "New AI startups"}, "search count"),
    ],
)
async def test_update_monitor_rejects_incomplete_changes(inputs, message):
    block = ExaUpdateMonitorBlock()
    api = AsyncMock()
    with patch.object(block, "_api", api), pytest.raises(ValueError, match=message):
        await collect_outputs(block, monitor_id=MONITOR_ID, **inputs)
    api.assert_not_awaited()


@pytest.mark.asyncio
async def test_monitor_run_reports_failure_reason():
    block = ExaGetWebsetMonitorRunBlock()
    run = sample_monitor_run(status="failed", failedReason="Out of credits")
    api = AsyncMock(return_value=run)
    with patch.object(block, "_api", api):
        outputs = dict(await collect_outputs(block, monitor_id="m/1", run_id="r1"))

    assert api.call_args.args[1:3] == ("GET", "/monitors/m%2F1/runs/r1")
    assert outputs["status"] == "failed"
    assert outputs["failed_reason"] == "Out of credits"


@pytest.mark.asyncio
async def test_team_info_with_unlimited_and_full_limits():
    block = ExaGetWebsetsTeamInfoBlock()
    unlimited = sample_team(limits={"maxConcurrent": None, "maxQueued": None})
    with patch.object(block, "_api", AsyncMock(return_value=unlimited)):
        outputs = dict(await collect_outputs(block))
    assert outputs["available_slots"] is None
    assert outputs["has_capacity"] is True

    full = sample_team(concurrency={"active": 12, "queued": 3})
    with patch.object(block, "_api", AsyncMock(return_value=full)):
        outputs = dict(await collect_outputs(block))
    assert outputs["available_slots"] == 0
    assert outputs["has_capacity"] is False


@pytest.mark.asyncio
async def test_update_import_requires_a_change():
    block = ExaUpdateImportBlock()
    api = AsyncMock()
    with patch.object(block, "_api", api), pytest.raises(
        ValueError, match="title or metadata"
    ):
        await collect_outputs(block, import_id=IMPORT_ID)
    api.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_import_patches_title_and_metadata():
    block = ExaUpdateImportBlock()
    api = AsyncMock(return_value={"id": IMPORT_ID, "status": "canceled"})
    with patch.object(block, "_api", api):
        outputs = dict(
            await collect_outputs(
                block, import_id=IMPORT_ID, title="Q3", metadata={"owner": "ops"}
            )
        )

    assert api.call_args.args[1:3] == ("PATCH", f"/imports/{IMPORT_ID}")
    assert api.call_args.kwargs["json"] == {"title": "Q3", "metadata": {"owner": "ops"}}
    assert outputs["status"] == "canceled"
