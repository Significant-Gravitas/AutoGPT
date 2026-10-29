"""Payload parsing tests for the Exa webhook trigger blocks."""

from backend.blocks.exa._search_monitor_samples import (
    SAMPLE_MONITOR,
    SAMPLE_MONITOR_ID,
    SAMPLE_RUN_COMPLETED_EVENT,
)
from backend.blocks.exa.search_monitor_trigger import ExaSearchMonitorTriggerBlock
from backend.blocks.exa.search_monitors_models import SearchMonitorModel
from backend.blocks.exa.webhook_blocks import (
    EXAMPLE_ITEM_CREATED_EVENT,
    ExaWebsetWebhookBlock,
    WebsetEventFilter,
)


async def _run(block, **inputs) -> dict:
    outputs: dict = {}
    async for name, value in block.run(block.input_schema(**inputs)):
        outputs.setdefault(name, value)
    return outputs


async def _run_webset(payload: dict, **inputs) -> dict:
    return await _run(
        ExaWebsetWebhookBlock(),
        credentials={"provider": "exa", "id": "c", "type": "api_key"},
        payload=payload,
        **inputs,
    )


class TestWebsetWebhookBlock:
    async def test_reads_exa_event_envelope(self):
        outputs = await _run_webset(EXAMPLE_ITEM_CREATED_EVENT)
        assert outputs == {
            "event_type": "webset.item.created",
            "event_id": EXAMPLE_ITEM_CREATED_EVENT["id"],
            "webset_id": "webset_01jz2h3k4m5n6p7q8r9s0t1u2x",
            "data": EXAMPLE_ITEM_CREATED_EVENT["data"],
            "timestamp": EXAMPLE_ITEM_CREATED_EVENT["createdAt"],
            "metadata": {"team": "research"},
        }

    async def test_webset_events_take_the_id_from_the_webset_itself(self):
        payload = {
            "id": "e2",
            "type": "webset.created",
            "data": {"id": "webset_9", "object": "webset"},
            "createdAt": "2026-10-28T00:00:00Z",
        }
        outputs = await _run_webset(payload)
        assert outputs["webset_id"] == "webset_9"

    async def test_events_for_other_websets_are_skipped(self):
        outputs = await _run_webset(
            EXAMPLE_ITEM_CREATED_EVENT, webset_id="webset_other"
        )
        assert outputs == {}

    async def test_events_naming_no_webset_pass_the_webset_filter(self):
        payload = {
            "id": "e3",
            "type": "import.completed",
            "data": {"id": "import_1", "object": "import"},
        }
        outputs = await _run_webset(payload, webset_id="webset_1")
        assert outputs["event_type"] == "import.completed"
        assert outputs["webset_id"] == "webset_1"

    async def test_disabled_events_are_skipped(self):
        payload = {**EXAMPLE_ITEM_CREATED_EVENT, "type": "webset.deleted"}
        outputs = await _run_webset(payload)
        assert outputs == {}

    async def test_monitor_events_fire_when_enabled(self):
        payload = {**EXAMPLE_ITEM_CREATED_EVENT, "type": "monitor.run.completed"}
        enabled = WebsetEventFilter(monitor_run_completed=True)
        outputs = await _run_webset(payload, event_filter=enabled)
        assert outputs["event_type"] == "monitor.run.completed"


class TestSearchMonitorTriggerBlock:
    async def test_run_completed_event_yields_results(self):
        block = ExaSearchMonitorTriggerBlock()
        outputs = []
        inputs = block.input_schema(payload=SAMPLE_RUN_COMPLETED_EVENT)
        async for name, value in block.run(inputs):
            outputs.append((name, value))
        names = [name for name, _ in outputs]
        assert names.count("result") == 1
        values = dict(outputs)
        assert values["monitor_id"] == SAMPLE_MONITOR_ID
        assert values["status"] == "completed"
        assert values["content"] == "One new facility was announced in Nevada."
        assert values["grounding"][0].citations[0].url == (
            "https://example.com/announcement"
        )
        assert values["metadata"] == {"workspace_id": "workspace_123"}
        assert "monitor" not in values

    async def test_failed_run_reports_reason(self):
        payload = {
            "id": "e4",
            "type": "monitor.run.completed",
            "data": {
                "id": "run_2",
                "monitorId": SAMPLE_MONITOR_ID,
                "status": "failed",
                "output": None,
                "failReason": "insufficient_credits",
            },
        }
        outputs = await _run(ExaSearchMonitorTriggerBlock(), payload=payload)
        assert outputs["status"] == "failed"
        assert outputs["fail_reason"] == "insufficient_credits"
        assert outputs["results"] == []
        assert "result" not in outputs

    async def test_lifecycle_event_yields_the_monitor(self):
        payload = {"id": "e5", "type": "monitor.updated", "data": SAMPLE_MONITOR}
        outputs = await _run(ExaSearchMonitorTriggerBlock(), payload=payload)
        assert outputs["monitor"] == SearchMonitorModel.from_api(SAMPLE_MONITOR)
        assert outputs["monitor_id"] == SAMPLE_MONITOR_ID
        assert outputs["status"] == "active"
        assert "run" not in outputs
