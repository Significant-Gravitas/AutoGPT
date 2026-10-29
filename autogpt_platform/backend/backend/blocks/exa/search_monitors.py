"""
Exa Search Monitor create block.

A Search Monitor runs an Exa search on a schedule, drops results earlier runs
already returned, and posts what is new to a webhook. Pair these blocks with
the "Exa Search Monitor Trigger" block, whose webhook URL goes in
``webhook_url`` and which receives each run's results.
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
from ._search_monitor_samples import (
    SAMPLE_CREATED_MONITOR,
    SAMPLE_MONITOR,
    SAMPLE_MONITOR_ID,
)
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .helpers import ContentSettings, merge_exa_cost
from .search_monitors_models import (
    MONITORS_URL,
    SearchMonitorApiMixin,
    SearchMonitorModel,
    SearchMonitorWebhookEvent,
    build_search,
    build_trigger,
    build_webhook,
)

WEBHOOK_URL_HELP = (
    "HTTPS URL Exa posts each run to. To handle runs in a workflow, add an "
    "'Exa Search Monitor Trigger' block to it, save, and paste that block's "
    "webhook URL here. Must be publicly reachable (no localhost or private IPs); "
    "redirects are not followed."
)
PERIOD_HELP = (
    "How often the monitor runs: a whole number followed by h (hours) or d "
    "(days), minimum 1h, e.g. 6h, 1d or 7d. The schedule is anchored to the "
    "creation time and a run may start up to 30 minutes late."
)
OUTPUT_SCHEMA_HELP = (
    'Format of each run\'s synthesized output. Defaults to {"type": "text"} (a '
    'plain-text summary). Use {"type": "object", "properties": {...}} for JSON '
    "matching your schema; citations are returned separately in grounding, so "
    "leave them out of the schema."
)


class ExaCreateSearchMonitorBlock(SearchMonitorApiMixin, Block):
    """Create a Search Monitor that runs an Exa search on a schedule."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        query: str = SchemaField(
            description="What to watch for. Describe the ongoing signal rather "
            "than adding dates: each run only reports results that are new.",
            placeholder="new battery recycling facilities announced in North America",
        )
        webhook_url: str = SchemaField(
            description=WEBHOOK_URL_HELP,
            placeholder="https://.../api/integrations/exa/webhooks/.../ingress",
        )
        period: str = SchemaField(
            default="1d",
            description=PERIOD_HELP + " Leave empty for a manual-only monitor "
            "that runs only when triggered.",
            placeholder="1d",
            advanced=False,
        )
        name: str = SchemaField(
            default="",
            description="Optional display name for the monitor",
            placeholder="Battery recycling expansion",
            advanced=False,
        )
        num_results: int = SchemaField(
            default=10,
            description="Search results to fetch per run (1-100)",
            ge=1,
            le=100,
        )
        include_domains: list[str] = SchemaField(
            default_factory=list,
            description="Only return results from these domains",
            placeholder="example.com",
        )
        exclude_domains: list[str] = SchemaField(
            default_factory=list,
            description="Never return results from these domains",
            placeholder="example.com",
        )
        contents: ContentSettings | None = SchemaField(
            default=None,
            description="Page content to attach to each result (text, highlights, "
            "summary, livecrawl and more)",
        )
        output_schema: dict | None = SchemaField(
            default=None, description=OUTPUT_SCHEMA_HELP
        )
        metadata: dict[str, str] = SchemaField(
            default_factory=dict,
            description="Key-value pairs echoed back in every webhook delivery, "
            "useful for routing results",
        )
        webhook_events: list[SearchMonitorWebhookEvent] = SchemaField(
            default_factory=list,
            description="Events to post to the webhook. Leave empty to receive "
            "all of them (lifecycle, run started and run completed).",
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(description="The new monitor's ID")
        webhook_secret: str = SchemaField(
            description="Secret for verifying this monitor's webhook deliveries. "
            "Exa shows it only once: paste it into the Exa Search Monitor "
            "Trigger block's webhook secret, or store it safely."
        )
        status: str = SchemaField(description="Monitor status, normally active")
        next_run_at: str | None = SchemaField(
            description="When the first scheduled run happens (ISO 8601); empty "
            "for a manual-only monitor"
        )
        monitor: SearchMonitorModel = SchemaField(
            title="Monitor", description="The full monitor"
        )

    def __init__(self):
        super().__init__(
            id="5ea02a85-c529-4060-b06f-7f702478dd5c",
            description="Create an Exa Search Monitor that re-runs a search on a "
            "schedule and posts only new results to a webhook (pair it with the "
            "Exa Search Monitor Trigger block)",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateSearchMonitorBlock.Input,
            output_schema=ExaCreateSearchMonitorBlock.Output,
            effect=BlockEffect.EXTERNAL,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": SAMPLE_MONITOR["search"]["query"],
                "webhook_url": "https://example.com/webhooks/exa",
                "name": "Battery recycling expansion",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("monitor_id", SAMPLE_MONITOR_ID),
                ("webhook_secret", SAMPLE_CREATED_MONITOR["webhookSecret"]),
                ("status", "active"),
                ("next_run_at", SAMPLE_MONITOR["nextRunAt"]),
                ("monitor", SearchMonitorModel.from_api(SAMPLE_MONITOR)),
            ],
            test_mock={"call_exa": lambda *args, **kwargs: SAMPLE_CREATED_MONITOR},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials, "POST", MONITORS_URL, body=build_create_body(input_data)
        )
        merge_exa_cost(self, data)
        monitor = SearchMonitorModel.from_api(data)
        yield "monitor_id", monitor.id
        yield "webhook_secret", data.get("webhookSecret", "")
        yield "status", monitor.status
        yield "next_run_at", monitor.next_run_at
        yield "monitor", monitor


def build_create_body(input_data: ExaCreateSearchMonitorBlock.Input) -> dict[str, Any]:
    """Build the createMonitor request body from the block inputs."""
    if not input_data.webhook_url.strip():
        raise ValueError("A webhook URL is required: Exa posts every run to it")
    body: dict[str, Any] = {
        "search": build_search(
            query=input_data.query,
            num_results=input_data.num_results,
            include_domains=input_data.include_domains,
            exclude_domains=input_data.exclude_domains,
            contents=input_data.contents,
        ),
        "webhook": build_webhook(input_data.webhook_url, input_data.webhook_events),
    }
    if input_data.name:
        body["name"] = input_data.name
    if input_data.period.strip():
        body["trigger"] = build_trigger(input_data.period)
    if input_data.output_schema:
        body["outputSchema"] = input_data.output_schema
    if input_data.metadata:
        body["metadata"] = input_data.metadata
    return body
