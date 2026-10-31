"""
Exa Search Monitor update block.
"""

from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._exa_rest import path_segment
from ._search_monitor_samples import SAMPLE_MONITOR, SAMPLE_MONITOR_ID
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .helpers import ContentSettings, merge_exa_cost
from .search_monitors import PERIOD_HELP, WEBHOOK_URL_HELP
from .search_monitors_models import (
    MONITORS_URL,
    SearchMonitorApiMixin,
    SearchMonitorModel,
    SearchMonitorSettableStatus,
    SearchMonitorWebhookEvent,
    build_search,
    build_trigger,
    build_webhook,
)


class ExaUpdateSearchMonitorBlock(SearchMonitorApiMixin, Block):
    """Change a Search Monitor's search, schedule, output, webhook or status."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The monitor to update", placeholder=SAMPLE_MONITOR_ID
        )
        status: SearchMonitorSettableStatus | None = SchemaField(
            default=None,
            description="active runs on schedule; paused runs only when "
            "triggered. Leave empty to keep the current status.",
            advanced=False,
        )
        query: str | None = SchemaField(
            default=None, description="New search query", advanced=False
        )
        period: str | None = SchemaField(
            default=None,
            description=PERIOD_HELP + " Leave empty to keep the current schedule.",
            placeholder="1d",
            advanced=False,
        )
        remove_schedule: bool = SchemaField(
            default=False,
            description="Remove the schedule so the monitor only runs when "
            "triggered (overrides the run interval)",
        )
        name: str | None = SchemaField(default=None, description="New name")
        num_results: int | None = SchemaField(
            default=None, description="Search results per run (1-100)", ge=1, le=100
        )
        include_domains: list[str] | None = SchemaField(
            default=None,
            description="Replace the included domains. Leave empty to keep the current list",
        )
        exclude_domains: list[str] | None = SchemaField(
            default=None,
            description="Replace the excluded domains. Leave empty to keep the current list",
        )
        contents: ContentSettings | None = SchemaField(
            default=None,
            description="Replace the per-result content options. Leave empty to keep the current ones",
        )
        output_schema: dict | None = SchemaField(
            default=None,
            description='Replace the output format: a JSON Schema, or {"type": "text"} for a written summary. Leave empty to keep the current one',
        )
        metadata: dict[str, str] | None = SchemaField(
            default=None, description="Replace the metadata"
        )
        clear_metadata: bool = SchemaField(
            default=False, description="Remove all metadata (overrides metadata)"
        )
        webhook_url: str | None = SchemaField(
            default=None, description="New webhook URL. " + WEBHOOK_URL_HELP
        )
        webhook_events: list[SearchMonitorWebhookEvent] | None = SchemaField(
            default=None,
            description="Replace the webhook's subscribed events. Leave empty to keep the current ones",
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(description="The monitor's ID")
        status: str = SchemaField(description="Status after the update")
        next_run_at: str | None = SchemaField(
            description="Next scheduled run (ISO 8601); empty if manual-only"
        )
        monitor: SearchMonitorModel = SchemaField(
            title="Monitor", description="The updated monitor"
        )

    def __init__(self):
        super().__init__(
            id="e8db3a38-ac76-48eb-a4a7-1ba405f483a4",
            description="Update an Exa Search Monitor: pause or resume it, or "
            "change its query, schedule, output format or webhook",
            categories={BlockCategory.SEARCH},
            input_schema=ExaUpdateSearchMonitorBlock.Input,
            output_schema=ExaUpdateSearchMonitorBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "monitor_id": SAMPLE_MONITOR_ID,
                "status": SearchMonitorSettableStatus.ACTIVE,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("monitor_id", SAMPLE_MONITOR_ID),
                ("status", "active"),
                ("next_run_at", SAMPLE_MONITOR["nextRunAt"]),
                ("monitor", SearchMonitorModel.from_api(SAMPLE_MONITOR)),
            ],
            test_mock={"call_exa": lambda *args, **kwargs: SAMPLE_MONITOR},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials,
            "PATCH",
            f"{MONITORS_URL}/{path_segment(input_data.monitor_id)}",
            body=build_update_body(input_data),
        )
        merge_exa_cost(self, data)
        monitor = SearchMonitorModel.from_api(data)
        yield "monitor_id", monitor.id
        yield "status", monitor.status
        yield "next_run_at", monitor.next_run_at
        yield "monitor", monitor


def build_update_body(input_data: ExaUpdateSearchMonitorBlock.Input) -> dict[str, Any]:
    """Build the updateMonitor body; only fields the user set are sent."""
    body: dict[str, Any] = {}
    if input_data.name is not None:
        body["name"] = input_data.name
    if input_data.status is not None:
        body["status"] = input_data.status.value
    search = build_search(
        query=input_data.query,
        num_results=input_data.num_results,
        include_domains=input_data.include_domains,
        exclude_domains=input_data.exclude_domains,
        contents=input_data.contents,
    )
    if search:
        body["search"] = search
    if input_data.remove_schedule:
        body["trigger"] = None
    elif input_data.period and input_data.period.strip():
        body["trigger"] = build_trigger(input_data.period)
    if input_data.output_schema:
        body["outputSchema"] = input_data.output_schema
    if input_data.clear_metadata:
        body["metadata"] = None
    elif input_data.metadata is not None:
        body["metadata"] = input_data.metadata
    webhook = build_webhook(input_data.webhook_url, input_data.webhook_events)
    if webhook:
        body["webhook"] = webhook
    if not body:
        raise ValueError("Nothing to update: set at least one field to change")
    return body
