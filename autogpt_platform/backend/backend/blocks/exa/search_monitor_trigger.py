"""
Trigger block fired by an Exa Search Monitor's webhook.

Setup: add this block to a workflow and save it to get its webhook URL, paste
that URL into "Exa Create Search Monitor" as the webhook URL, then paste the
monitor's one-time ``webhook_secret`` output back into this block so every
delivery's ``Exa-Signature`` is checked (see ``_webhook.py``).
"""

from typing import Any

from backend.sdk import (
    BaseModel,
    Block,
    BlockCategory,
    BlockManualWebhookConfig,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    Field,
    ProviderName,
    SchemaField,
)

from ._search_monitor_samples import (
    SAMPLE_MONITOR_ID,
    SAMPLE_RUN,
    SAMPLE_RUN_COMPLETED_EVENT,
    SAMPLE_RUN_ID,
)
from ._webhook import ExaWebhookManager
from .search_monitors_models import (
    MonitorGrounding,
    SearchMonitorModel,
    SearchMonitorRunModel,
)

_RUN_EVENT_PREFIX = "monitor.run."
_SAMPLE_RUN_MODEL = SearchMonitorRunModel.from_api(SAMPLE_RUN)


class SearchMonitorEventFilter(BaseModel):
    """Which Search Monitor webhook events start the workflow."""

    monitor_run_completed: bool = Field(
        default=True,
        description="A run finished (carries its results, or why it failed)",
    )
    monitor_run_created: bool = Field(default=False, description="A run started")
    monitor_created: bool = Field(default=False, description="The monitor was created")
    monitor_updated: bool = Field(
        default=False, description="The monitor's settings changed"
    )
    monitor_deleted: bool = Field(default=False, description="The monitor was deleted")


class ExaSearchMonitorTriggerBlock(Block):
    """Starts a workflow each time an Exa Search Monitor posts to this webhook."""

    class Input(BlockSchemaInput):
        payload: dict = SchemaField(hidden=True, default_factory=dict)
        # Read by name in _webhook.py (SEARCH_MONITOR_SECRET_INPUT).
        webhook_secret: str | None = SchemaField(
            title="Webhook secret",
            description="The webhook_secret output of Exa Create Search Monitor "
            "(Exa shows it only once). When set, deliveries without a valid "
            "Exa signature are rejected. Strongly recommended: until it is set, "
            "anyone who knows this block's URL can send it fake events.",
            default=None,
            secret=True,
            advanced=False,
        )
        events: SearchMonitorEventFilter = SchemaField(
            title="Events",
            description="Which monitor events start the workflow",
            default_factory=SearchMonitorEventFilter,
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(description="The raw event Exa delivered")
        event_type: str = SchemaField(
            description="The event, e.g. monitor.run.completed"
        )
        event_id: str = SchemaField(description="Unique ID of this event")
        created_at: str = SchemaField(description="When the event happened (ISO 8601)")
        monitor_id: str = SchemaField(description="The monitor the event is about")
        run_id: str = SchemaField(description="The run's ID (run events only)")
        status: str = SchemaField(
            description="Run status for run events (completed, failed, ...), "
            "or monitor status for monitor events"
        )
        results: list[dict[str, Any]] = SchemaField(
            description="New search results found by the run"
        )
        result: dict[str, Any] = SchemaField(description="Each result, one at a time")
        content: Any = SchemaField(
            description="Synthesized findings: text, or JSON matching the "
            "monitor's output schema"
        )
        grounding: list[MonitorGrounding] = SchemaField(
            description="Citations and confidence for each synthesized field"
        )
        fail_reason: str | None = SchemaField(
            description="Why the run failed; empty unless it failed"
        )
        run: SearchMonitorRunModel = SchemaField(
            title="Run", description="The full run"
        )
        monitor: SearchMonitorModel = SchemaField(
            title="Monitor",
            description="The monitor (monitor created/updated/deleted events only)",
        )
        metadata: dict[str, str] = SchemaField(
            description="The monitor's metadata, echoed back by Exa"
        )

    def __init__(self):
        super().__init__(
            id="6741c2a7-283e-4b3a-a912-9cfd0f66785d",
            description="Triggers when an Exa Search Monitor reports new results. "
            "Paste this block's webhook URL into Exa Create Search Monitor, then "
            "paste the monitor's webhook secret here.",
            categories={BlockCategory.INPUT, BlockCategory.SEARCH},
            input_schema=ExaSearchMonitorTriggerBlock.Input,
            output_schema=ExaSearchMonitorTriggerBlock.Output,
            webhook_config=BlockManualWebhookConfig(
                provider=ProviderName("exa"),
                webhook_type=ExaWebhookManager.WebhookType.SEARCH_MONITOR.value,
                event_filter_input="events",
            ),
            test_input={"payload": SAMPLE_RUN_COMPLETED_EVENT},
            test_output=[
                ("payload", SAMPLE_RUN_COMPLETED_EVENT),
                ("event_type", "monitor.run.completed"),
                ("event_id", SAMPLE_RUN_COMPLETED_EVENT["id"]),
                ("created_at", SAMPLE_RUN_COMPLETED_EVENT["createdAt"]),
                ("monitor_id", SAMPLE_MONITOR_ID),
                ("run_id", SAMPLE_RUN_ID),
                ("status", "completed"),
                ("results", _SAMPLE_RUN_MODEL.results),
                ("result", _SAMPLE_RUN_MODEL.results[0]),
                ("content", _SAMPLE_RUN_MODEL.content),
                ("grounding", _SAMPLE_RUN_MODEL.grounding),
                ("fail_reason", None),
                ("run", _SAMPLE_RUN_MODEL),
                ("metadata", {"workspace_id": "workspace_123"}),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        event_type = str(payload.get("type") or "")
        data: dict[str, Any] = payload.get("data") or {}

        yield "payload", payload
        yield "event_type", event_type
        yield "event_id", str(payload.get("id") or "")
        yield "created_at", str(payload.get("createdAt") or "")

        if event_type.startswith(_RUN_EVENT_PREFIX):
            for output in _run_outputs(SearchMonitorRunModel.from_api(data)):
                yield output
        else:
            monitor = SearchMonitorModel.from_api(data)
            yield "monitor_id", monitor.id
            yield "status", monitor.status
            yield "monitor", monitor

        yield "metadata", data.get("metadata") or {}


def _run_outputs(run: SearchMonitorRunModel) -> list[tuple[str, Any]]:
    """Outputs for a monitor.run.* event, in emission order."""
    return [
        ("monitor_id", run.monitor_id),
        ("run_id", run.id),
        ("status", run.status),
        ("results", run.results),
        *(("result", result) for result in run.results),
        ("content", run.content),
        ("grounding", run.grounding),
        ("fail_reason", run.fail_reason),
        ("run", run),
    ]
