"""
Exa Search Monitor blocks: get, list and delete.
"""

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
from ._exa_rest import QueryParams, add_param, path_segment
from ._search_monitor_samples import SAMPLE_MONITOR, SAMPLE_MONITOR_ID
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .helpers import merge_exa_cost
from .search_monitors_models import (
    MONITORS_URL,
    SearchMonitorApiMixin,
    SearchMonitorModel,
    SearchMonitorStatus,
)

_SAMPLE_MODEL = SearchMonitorModel.from_api(SAMPLE_MONITOR)


class ExaGetSearchMonitorBlock(SearchMonitorApiMixin, Block):
    """Fetch one Search Monitor's configuration and schedule."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The monitor to fetch", placeholder=SAMPLE_MONITOR_ID
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(description="The monitor's ID")
        status: str = SchemaField(description="active, paused or disabled")
        next_run_at: str | None = SchemaField(
            description="Next scheduled run (ISO 8601); empty if manual-only"
        )
        monitor: SearchMonitorModel = SchemaField(
            title="Monitor", description="The full monitor"
        )

    def __init__(self):
        super().__init__(
            id="2867f7c4-cff9-4559-b7bf-9ced299ca50e",
            description="Get an Exa Search Monitor's query, schedule, webhook and status",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetSearchMonitorBlock.Input,
            output_schema=ExaGetSearchMonitorBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "monitor_id": SAMPLE_MONITOR_ID,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("monitor_id", SAMPLE_MONITOR_ID),
                ("status", "active"),
                ("next_run_at", SAMPLE_MONITOR["nextRunAt"]),
                ("monitor", _SAMPLE_MODEL),
            ],
            test_mock={"call_exa": lambda *args, **kwargs: SAMPLE_MONITOR},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials, "GET", f"{MONITORS_URL}/{path_segment(input_data.monitor_id)}"
        )
        merge_exa_cost(self, data)
        monitor = SearchMonitorModel.from_api(data)
        yield "monitor_id", monitor.id
        yield "status", monitor.status
        yield "next_run_at", monitor.next_run_at
        yield "monitor", monitor


class ExaListSearchMonitorsBlock(SearchMonitorApiMixin, Block):
    """List the team's Search Monitors, optionally filtered."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        status: SearchMonitorStatus | None = SchemaField(
            default=None,
            description="Only list monitors in this status",
            advanced=False,
        )
        name: str = SchemaField(
            default="",
            description="Only list monitors whose name contains this text "
            "(case-insensitive)",
            advanced=False,
        )
        metadata: dict[str, str] = SchemaField(
            default_factory=dict,
            description="Only list monitors whose metadata has all of these "
            "key-value pairs (exact match)",
        )
        limit: int = SchemaField(
            default=50, description="Monitors per page (1-100)", ge=1, le=100
        )
        cursor: str = SchemaField(
            default="", description="Cursor from a previous page's next_cursor"
        )

    class Output(BlockSchemaOutput):
        monitors: list[SearchMonitorModel] = SchemaField(
            description="Monitors on this page"
        )
        monitor: SearchMonitorModel = SchemaField(
            title="Monitor", description="Each monitor, one at a time"
        )
        has_more: bool = SchemaField(description="Whether another page exists")
        next_cursor: str | None = SchemaField(
            description="Pass as cursor to fetch the next page"
        )

    def __init__(self):
        super().__init__(
            id="60c1f7cf-1f66-4c0d-877c-7e96298657ef",
            description="List Exa Search Monitors, filtered by status, name or metadata",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListSearchMonitorsBlock.Input,
            output_schema=ExaListSearchMonitorsBlock.Output,
            effect=BlockEffect.READ,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("monitors", [_SAMPLE_MODEL]),
                ("monitor", _SAMPLE_MODEL),
                ("has_more", False),
                ("next_cursor", None),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "data": [SAMPLE_MONITOR],
                    "hasMore": False,
                    "nextCursor": None,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials, "GET", MONITORS_URL, params=list_params(input_data)
        )
        monitors = [SearchMonitorModel.from_api(item) for item in data.get("data", [])]
        yield "monitors", monitors
        for monitor in monitors:
            yield "monitor", monitor
        yield "has_more", data.get("hasMore", False)
        yield "next_cursor", data.get("nextCursor")


def list_params(input_data: ExaListSearchMonitorsBlock.Input) -> QueryParams:
    """Query string for listMonitors; metadata uses ``metadata[key]=value``."""
    params: QueryParams = []
    add_param(params, "status", input_data.status.value if input_data.status else None)
    add_param(params, "name", input_data.name)
    add_param(params, "limit", input_data.limit)
    add_param(params, "cursor", input_data.cursor)
    for key, value in input_data.metadata.items():
        add_param(params, f"metadata[{key}]", value)
    return params


class ExaDeleteSearchMonitorBlock(SearchMonitorApiMixin, Block):
    """Permanently delete a Search Monitor."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The monitor to delete. This cannot be undone.",
            placeholder=SAMPLE_MONITOR_ID,
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(description="ID of the deleted monitor")
        monitor: SearchMonitorModel = SchemaField(
            title="Monitor", description="The monitor as it was when deleted"
        )

    def __init__(self):
        super().__init__(
            id="0045649e-3aed-4d25-aba8-b00cff03f32c",
            description="Permanently delete an Exa Search Monitor, stopping its runs",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteSearchMonitorBlock.Input,
            output_schema=ExaDeleteSearchMonitorBlock.Output,
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "monitor_id": SAMPLE_MONITOR_ID,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("monitor_id", SAMPLE_MONITOR_ID),
                ("monitor", _SAMPLE_MODEL),
            ],
            test_mock={"call_exa": lambda *args, **kwargs: SAMPLE_MONITOR},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials,
            "DELETE",
            f"{MONITORS_URL}/{path_segment(input_data.monitor_id)}",
        )
        monitor = SearchMonitorModel.from_api(data)
        yield "monitor_id", monitor.id or input_data.monitor_id
        yield "monitor", monitor
