"""
Exa Search Monitor batch action block: delete, pause or unpause by filter.
"""

from enum import Enum
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
from ._search_monitor_samples import SAMPLE_MONITOR_ID
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .search_monitors_models import (
    MONITORS_URL,
    SearchMonitorApiMixin,
    SearchMonitorStatus,
)


class SearchMonitorBatchAction(str, Enum):
    DELETE = "delete"  # permanent
    PAUSE = "pause"
    UNPAUSE = "unpause"


class ExaBatchSearchMonitorsBlock(SearchMonitorApiMixin, Block):
    """Delete, pause or unpause every Search Monitor matching a filter."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        action: SearchMonitorBatchAction = SchemaField(
            title="Action",
            description="delete removes matching monitors permanently; pause "
            "stops their scheduled runs; unpause makes them active again",
        )
        name: str = SchemaField(
            default="",
            description="Match monitors whose name contains this text "
            "(case-insensitive)",
            advanced=False,
        )
        status: SearchMonitorStatus | None = SchemaField(
            default=None, description="Match monitors in this status", advanced=False
        )
        metadata: dict[str, str] = SchemaField(
            default_factory=dict,
            description="Match monitors whose metadata has all of these pairs",
        )
        dry_run: bool = SchemaField(
            default=True,
            description="Only preview which monitors would be affected. Turn off "
            "to actually apply the action.",
            advanced=False,
        )
        limit: int = SchemaField(
            default=50,
            description="Most monitors to process in one call (1-500). Run again "
            "while has_more is true to process the rest.",
            ge=1,
            le=500,
        )

    class Output(BlockSchemaOutput):
        action: str = SchemaField(description="The action requested")
        affected: int = SchemaField(
            description="How many monitors were (or, in a dry run, would be) affected"
        )
        monitor_ids: list[str] = SchemaField(description="IDs of those monitors")
        dry_run: bool = SchemaField(description="Whether this was only a preview")
        has_more: bool = SchemaField(
            description="More monitors match; run again to process them"
        )

    def __init__(self):
        super().__init__(
            id="cc6a8cf0-583d-461d-a8e6-5400ff8b637c",
            description="Delete, pause or unpause every Exa Search Monitor matching "
            "a name, status or metadata filter (previews by default)",
            categories={BlockCategory.SEARCH},
            input_schema=ExaBatchSearchMonitorsBlock.Input,
            output_schema=ExaBatchSearchMonitorsBlock.Output,
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "action": SearchMonitorBatchAction.PAUSE,
                "status": SearchMonitorStatus.ACTIVE,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("action", "pause"),
                ("affected", 1),
                ("monitor_ids", [SAMPLE_MONITOR_ID]),
                ("dry_run", True),
                ("has_more", False),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "action": "pause",
                    "affected": 1,
                    "ids": [SAMPLE_MONITOR_ID],
                    "dry_run": True,
                    "has_more": False,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials, "POST", f"{MONITORS_URL}/batch", body=batch_body(input_data)
        )
        yield "action", data.get("action", input_data.action.value)
        yield "affected", data.get("affected", 0)
        yield "monitor_ids", data.get("ids", [])
        yield "dry_run", data.get("dry_run", input_data.dry_run)
        yield "has_more", data.get("has_more", False)


def batch_body(input_data: ExaBatchSearchMonitorsBlock.Input) -> dict[str, Any]:
    """Build the batchMonitors body; Exa requires at least one filter."""
    monitor_filter: dict[str, Any] = {}
    if input_data.name:
        monitor_filter["name"] = input_data.name
    if input_data.status is not None:
        monitor_filter["status"] = input_data.status.value
    if input_data.metadata:
        monitor_filter["metadata"] = input_data.metadata
    if not monitor_filter:
        raise ValueError(
            "Set a name, status or metadata filter: batch actions never apply "
            "to all monitors at once"
        )
    return {
        "action": input_data.action.value,
        "filter": monitor_filter,
        "dry_run": input_data.dry_run,
        "limit": input_data.limit,
    }
