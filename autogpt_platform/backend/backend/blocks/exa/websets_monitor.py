"""
Exa Websets Monitor Management Blocks

This module provides blocks for creating and managing monitors that automatically
keep websets updated with fresh data on a schedule.
"""

from enum import Enum
from typing import Optional

from exa_py import AsyncExa
from exa_py.websets.types import Monitor as SdkMonitor
from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT


# Mirrored model for stability - don't use SDK types directly in block outputs
class MonitorModel(BaseModel):
    """Stable output model mirroring SDK Monitor."""

    id: str
    status: str
    webset_id: str
    behavior_type: str
    behavior_config: dict
    cron_expression: str
    timezone: str
    next_run_at: str
    last_run: dict
    metadata: dict
    created_at: str
    updated_at: str

    @classmethod
    def from_sdk(cls, monitor: SdkMonitor) -> "MonitorModel":
        """Convert SDK Monitor to our stable model."""
        # Extract behavior information
        behavior_dict = monitor.behavior.model_dump(by_alias=True, exclude_none=True)
        behavior_type = behavior_dict.get("type", "unknown")
        behavior_config = behavior_dict.get("config", {})

        # Extract cadence information
        cadence_dict = monitor.cadence.model_dump(by_alias=True, exclude_none=True)
        cron_expr = cadence_dict.get("cron", "")
        timezone = cadence_dict.get("timezone", "Etc/UTC")

        # Extract last run information
        last_run_dict = {}
        if monitor.last_run:
            last_run_dict = monitor.last_run.model_dump(
                by_alias=True, exclude_none=True
            )

        # Handle status enum
        status_str = (
            monitor.status.value
            if hasattr(monitor.status, "value")
            else str(monitor.status)
        )

        return cls(
            id=monitor.id,
            status=status_str,
            webset_id=monitor.webset_id,
            behavior_type=behavior_type,
            behavior_config=behavior_config,
            cron_expression=cron_expr,
            timezone=timezone,
            next_run_at=monitor.next_run_at.isoformat() if monitor.next_run_at else "",
            last_run=last_run_dict,
            metadata=monitor.metadata or {},
            created_at=monitor.created_at.isoformat() if monitor.created_at else "",
            updated_at=monitor.updated_at.isoformat() if monitor.updated_at else "",
        )


class MonitorStatus(str, Enum):
    """Status of a monitor."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    PAUSED = "paused"


class MonitorBehaviorType(str, Enum):
    """Type of behavior for a monitor."""

    SEARCH = "search"  # Run new searches
    REFRESH = "refresh"  # Refresh existing items


class SearchBehavior(str, Enum):
    """How search results interact with existing items."""

    APPEND = "append"
    OVERRIDE = "override"


class ExaCreateMonitorBlock(Block):
    """Create a monitor to automatically keep a webset updated on a schedule."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to monitor",
            placeholder="webset-id-or-external-id",
        )

        # Schedule configuration
        cron_expression: str = SchemaField(
            description="Cron expression for scheduling (5 fields, max once per day)",
            placeholder="0 9 * * 1",  # Every Monday at 9 AM
        )
        timezone: str = SchemaField(
            default="Etc/UTC",
            description="IANA timezone for the schedule",
            placeholder="America/New_York",
            advanced=True,
        )

        # Behavior configuration
        behavior_type: MonitorBehaviorType = SchemaField(
            default=MonitorBehaviorType.SEARCH,
            description="Type of monitor behavior (search for new items or refresh existing)",
        )

        # Search configuration (for SEARCH behavior)
        search_query: Optional[str] = SchemaField(
            default=None,
            description="Search query for finding new items (required for search behavior)",
            placeholder="AI startups that raised funding in the last week",
        )
        search_count: int = SchemaField(
            default=10,
            description="Number of items to find in each search",
            ge=1,
            le=100,
        )
        search_criteria: list[str] = SchemaField(
            default_factory=list,
            description="Criteria that items must meet",
            advanced=True,
        )
        search_behavior: SearchBehavior = SchemaField(
            default=SearchBehavior.APPEND,
            description="How new results interact with existing items",
            advanced=True,
        )
        entity_type: Optional[str] = SchemaField(
            default=None,
            description="Type of entity to search for (company, person, etc.)",
            advanced=True,
        )

        # Refresh configuration (for REFRESH behavior)
        refresh_content: bool = SchemaField(
            default=True,
            description="Refresh content from source URLs (for refresh behavior)",
            advanced=True,
        )
        refresh_enrichments: bool = SchemaField(
            default=True,
            description="Re-run enrichments on items (for refresh behavior)",
            advanced=True,
        )

        # Metadata
        metadata: Optional[dict] = SchemaField(
            default=None,
            description="Metadata to attach to the monitor",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(
            description="The unique identifier for the created monitor"
        )
        webset_id: str = SchemaField(description="The webset this monitor belongs to")
        status: str = SchemaField(description="Status of the monitor")
        behavior_type: str = SchemaField(description="Type of monitor behavior")
        next_run_at: Optional[str] = SchemaField(
            description="When the monitor will next run"
        )
        cron_expression: str = SchemaField(description="The schedule cron expression")
        timezone: str = SchemaField(description="The timezone for scheduling")
        created_at: str = SchemaField(description="When the monitor was created")

    def __init__(self):
        super().__init__(
            id="f8a9b0c1-d2e3-4567-890a-bcdef1234567",
            description="Create automated monitors to keep websets updated with fresh data on a schedule",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateMonitorBlock.Input,
            output_schema=ExaCreateMonitorBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "webset_id": "test-webset",
                "cron_expression": "0 9 * * 1",
                "behavior_type": MonitorBehaviorType.SEARCH,
                "search_query": "AI startups",
                "search_count": 10,
            },
            test_output=[
                ("monitor_id", "monitor-123"),
                ("webset_id", "test-webset"),
                ("status", "enabled"),
                ("behavior_type", "search"),
                ("next_run_at", "2024-01-01T00:00:00"),
                ("cron_expression", "0 9 * * 1"),
                ("timezone", "Etc/UTC"),
                ("created_at", "2024-01-01T00:00:00"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock=self._create_test_mock(),
        )

    @staticmethod
    def _create_test_mock():
        """Create test mocks for the AsyncExa SDK."""
        from datetime import datetime
        from unittest.mock import MagicMock

        # Create mock SDK monitor object
        mock_monitor = MagicMock()
        mock_monitor.id = "monitor-123"
        mock_monitor.status = MagicMock(value="enabled")
        mock_monitor.webset_id = "test-webset"
        mock_monitor.next_run_at = datetime.fromisoformat("2024-01-01T00:00:00")
        mock_monitor.created_at = datetime.fromisoformat("2024-01-01T00:00:00")
        mock_monitor.updated_at = datetime.fromisoformat("2024-01-01T00:00:00")
        mock_monitor.metadata = {}
        mock_monitor.last_run = None

        # Mock behavior
        mock_behavior = MagicMock()
        mock_behavior.model_dump = MagicMock(
            return_value={"type": "search", "config": {}}
        )
        mock_monitor.behavior = mock_behavior

        # Mock cadence
        mock_cadence = MagicMock()
        mock_cadence.model_dump = MagicMock(
            return_value={"cron": "0 9 * * 1", "timezone": "Etc/UTC"}
        )
        mock_monitor.cadence = mock_cadence

        return {
            "_get_client": lambda *args, **kwargs: MagicMock(
                websets=MagicMock(
                    monitors=MagicMock(create=lambda *args, **kwargs: mock_monitor)
                )
            )
        }

    def _get_client(self, api_key: str) -> AsyncExa:
        """Get Exa client (separated for testing)."""
        return AsyncExa(api_key=api_key)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = self._get_client(credentials.api_key.get_secret_value())

        # Build the payload
        payload = {
            "websetId": input_data.webset_id,
            "cadence": {
                "cron": input_data.cron_expression,
                "timezone": input_data.timezone,
            },
        }

        # Build behavior configuration based on type
        if input_data.behavior_type == MonitorBehaviorType.SEARCH:
            behavior_config = {
                "query": input_data.search_query or "",
                "count": input_data.search_count,
                "behavior": input_data.search_behavior.value,
            }

            if input_data.search_criteria:
                behavior_config["criteria"] = [
                    {"description": c} for c in input_data.search_criteria
                ]

            if input_data.entity_type:
                behavior_config["entity"] = {"type": input_data.entity_type}

            payload["behavior"] = {
                "type": "search",
                "config": behavior_config,
            }
        else:
            # REFRESH behavior
            payload["behavior"] = {
                "type": "refresh",
                "config": {
                    "content": input_data.refresh_content,
                    "enrichments": input_data.refresh_enrichments,
                },
            }

        # Add metadata if provided
        if input_data.metadata:
            payload["metadata"] = input_data.metadata

        sdk_monitor = aexa.websets.monitors.create(params=payload)

        monitor = MonitorModel.from_sdk(sdk_monitor)

        # Yield all fields
        yield "monitor_id", monitor.id
        yield "webset_id", monitor.webset_id
        yield "status", monitor.status
        yield "behavior_type", monitor.behavior_type
        yield "next_run_at", monitor.next_run_at
        yield "cron_expression", monitor.cron_expression
        yield "timezone", monitor.timezone
        yield "created_at", monitor.created_at


class ExaGetMonitorBlock(Block):
    """Get the details and status of a monitor."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The ID of the monitor to retrieve",
            placeholder="monitor-id",
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(
            description="The unique identifier for the monitor"
        )
        webset_id: str = SchemaField(description="The webset this monitor belongs to")
        status: str = SchemaField(description="Current status of the monitor")
        behavior_type: str = SchemaField(description="Type of monitor behavior")
        behavior_config: dict = SchemaField(
            description="Configuration for the monitor behavior"
        )
        cron_expression: str = SchemaField(description="The schedule cron expression")
        timezone: str = SchemaField(description="The timezone for scheduling")
        next_run_at: Optional[str] = SchemaField(
            description="When the monitor will next run"
        )
        last_run: Optional[dict] = SchemaField(
            description="Information about the last run"
        )
        created_at: str = SchemaField(description="When the monitor was created")
        updated_at: str = SchemaField(description="When the monitor was last updated")
        metadata: dict = SchemaField(description="Metadata attached to the monitor")

    def __init__(self):
        super().__init__(
            id="5c852a2d-d505-4a56-b711-7def8dd14e72",
            description="Get the details and status of a webset monitor",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetMonitorBlock.Input,
            output_schema=ExaGetMonitorBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        sdk_monitor = aexa.websets.monitors.get(monitor_id=input_data.monitor_id)

        monitor = MonitorModel.from_sdk(sdk_monitor)

        # Yield all fields
        yield "monitor_id", monitor.id
        yield "webset_id", monitor.webset_id
        yield "status", monitor.status
        yield "behavior_type", monitor.behavior_type
        yield "behavior_config", monitor.behavior_config
        yield "cron_expression", monitor.cron_expression
        yield "timezone", monitor.timezone
        yield "next_run_at", monitor.next_run_at
        yield "last_run", monitor.last_run
        yield "created_at", monitor.created_at
        yield "updated_at", monitor.updated_at
        yield "metadata", monitor.metadata


class ExaUpdateMonitorBlock(Block):
    """Update a monitor's configuration."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The ID of the monitor to update",
            placeholder="monitor-id",
        )
        status: Optional[MonitorStatus] = SchemaField(
            default=None,
            description="New status for the monitor",
        )
        cron_expression: Optional[str] = SchemaField(
            default=None,
            description="New cron expression for scheduling",
        )
        timezone: Optional[str] = SchemaField(
            default=None,
            description="New timezone for the schedule",
            advanced=True,
        )
        metadata: Optional[dict] = SchemaField(
            default=None,
            description="New metadata for the monitor",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(
            description="The unique identifier for the monitor"
        )
        status: str = SchemaField(description="Updated status of the monitor")
        next_run_at: Optional[str] = SchemaField(
            description="When the monitor will next run"
        )
        updated_at: str = SchemaField(description="When the monitor was updated")
        success: str = SchemaField(description="Whether the update was successful")

    def __init__(self):
        super().__init__(
            id="245102c3-6af3-4515-a308-c2210b7939d2",
            description="Update a monitor's status, schedule, or metadata",
            categories={BlockCategory.SEARCH},
            input_schema=ExaUpdateMonitorBlock.Input,
            output_schema=ExaUpdateMonitorBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # Build update payload
        payload = {}

        if input_data.status is not None:
            payload["status"] = input_data.status.value

        if input_data.cron_expression is not None or input_data.timezone is not None:
            cadence = {}
            if input_data.cron_expression:
                cadence["cron"] = input_data.cron_expression
            if input_data.timezone:
                cadence["timezone"] = input_data.timezone
            payload["cadence"] = cadence

        if input_data.metadata is not None:
            payload["metadata"] = input_data.metadata

        sdk_monitor = aexa.websets.monitors.update(
            monitor_id=input_data.monitor_id, params=payload
        )

        # Convert to our stable model
        monitor = MonitorModel.from_sdk(sdk_monitor)

        # Yield fields
        yield "monitor_id", monitor.id
        yield "status", monitor.status
        yield "next_run_at", monitor.next_run_at
        yield "updated_at", monitor.updated_at
        yield "success", "true"


class ExaDeleteMonitorBlock(Block):
    """Delete a monitor from a webset."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The ID of the monitor to delete",
            placeholder="monitor-id",
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(description="The ID of the deleted monitor")
        success: str = SchemaField(description="Whether the deletion was successful")

    def __init__(self):
        super().__init__(
            id="f16f9b10-0c4d-4db8-997d-7b96b6026094",
            description="Delete a monitor from a webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteMonitorBlock.Input,
            output_schema=ExaDeleteMonitorBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        deleted_monitor = aexa.websets.monitors.delete(monitor_id=input_data.monitor_id)

        yield "monitor_id", deleted_monitor.id
        yield "success", "true"


class ExaListMonitorsBlock(Block):
    """List all monitors with pagination."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: Optional[str] = SchemaField(
            default=None,
            description="Filter monitors by webset ID",
            placeholder="webset-id",
        )
        limit: int = SchemaField(
            default=25,
            description="Number of monitors to return",
            ge=1,
            le=100,
        )
        cursor: Optional[str] = SchemaField(
            default=None,
            description="Cursor for pagination",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        monitors: list[dict] = SchemaField(description="List of monitors")
        monitor: dict = SchemaField(
            description="Individual monitor (yielded for each monitor)"
        )
        has_more: bool = SchemaField(
            description="Whether there are more monitors to paginate through"
        )
        next_cursor: Optional[str] = SchemaField(
            description="Cursor for the next page of results"
        )

    def __init__(self):
        super().__init__(
            id="f06e2b38-5397-4e8f-aa85-491149dd98df",
            description="List all monitors with optional webset filtering",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListMonitorsBlock.Input,
            output_schema=ExaListMonitorsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        response = aexa.websets.monitors.list(
            cursor=input_data.cursor,
            limit=input_data.limit,
            webset_id=input_data.webset_id,
        )

        # Convert SDK monitors to our stable models
        monitors = [MonitorModel.from_sdk(m) for m in response.data]

        # Yield the full list
        yield "monitors", [m.model_dump() for m in monitors]

        # Yield individual monitors for graph chaining
        for monitor in monitors:
            yield "monitor", monitor.model_dump()

        # Yield pagination metadata
        yield "has_more", response.has_more
        yield "next_cursor", response.next_cursor
