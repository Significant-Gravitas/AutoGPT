"""
Exa Websets Monitor Management Blocks

This module provides blocks for creating and managing monitors that automatically
keep websets updated with fresh data on a schedule.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .websets_api import ExaWebsetsBlockBase, segment
from .websets_fixtures import MONITOR_ID, WEBSET_ID, sample_monitor
from .websets_payloads import criteria_payload, entity_payload


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
    def from_api(cls, monitor: dict[str, Any]) -> "MonitorModel":
        """Build the stable model from a raw Websets API monitor object."""
        behavior = monitor.get("behavior") or {}
        cadence = monitor.get("cadence") or {}
        return cls(
            id=monitor.get("id", ""),
            status=monitor.get("status", ""),
            webset_id=monitor.get("websetId", ""),
            behavior_type=behavior.get("type", "unknown"),
            behavior_config=behavior.get("config") or {},
            cron_expression=cadence.get("cron", ""),
            timezone=cadence.get("timezone") or "Etc/UTC",
            next_run_at=monitor.get("nextRunAt") or "",
            last_run=monitor.get("lastRun") or {},
            metadata=monitor.get("metadata") or {},
            created_at=monitor.get("createdAt") or "",
            updated_at=monitor.get("updatedAt") or "",
        )


class MonitorStatus(str, Enum):
    """Status of a monitor."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    PAUSED = "paused"  # Legacy: Exa no longer has 'paused'; sent as 'disabled'


class MonitorBehaviorType(str, Enum):
    """Type of behavior for a monitor."""

    SEARCH = "search"  # Run new searches
    REFRESH = "refresh"  # Legacy: Exa removed refresh monitors; rejected with an error


class SearchBehavior(str, Enum):
    """How search results interact with existing items."""

    APPEND = "append"
    OVERRIDE = "override"


class ExaCreateMonitorBlock(ExaWebsetsBlockBase):
    """Create a monitor to automatically keep a webset updated on a schedule."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID of the Webset to monitor",
            placeholder="webset-id",
        )

        # Schedule configuration
        cron_expression: str = SchemaField(
            description="When to run, as a 5-field Unix cron expression. Exa allows at most one run per day.",
            placeholder="0 9 * * 1",  # Every Monday at 9 AM
        )
        timezone: str = SchemaField(
            default="Etc/UTC",
            description="IANA timezone the cron expression is read in",
            placeholder="America/New_York",
            advanced=True,
        )

        # Behavior configuration
        behavior_type: MonitorBehaviorType = SchemaField(
            default=MonitorBehaviorType.SEARCH,
            description="What each run does. 'search' finds new items; 'refresh' is legacy (Exa removed it) and fails with an error.",
        )

        # Search configuration (for SEARCH behavior)
        search_query: str | None = SchemaField(
            default=None,
            description="Query each run searches for (2-10,000 characters). Leave empty to reuse the webset's last search query.",
            placeholder="AI startups that raised funding in the last week",
        )
        search_count: int = SchemaField(
            default=10,
            description="Maximum number of new items to find per run",
            ge=1,
            le=100,
        )
        search_criteria: list[str] = SchemaField(
            default_factory=list,
            description="Up to 5 criteria new items must meet. Leave empty to reuse the last search's criteria.",
            advanced=True,
        )
        search_behavior: SearchBehavior = SchemaField(
            default=SearchBehavior.APPEND,
            description="'append' adds new items to the webset; 'override' re-evaluates existing items against the run's criteria",
            advanced=True,
        )
        entity_type: str | None = SchemaField(
            default=None,
            description="Kind of item to find: company, person, article, research_paper or custom. Leave empty to reuse the last search's entity.",
            advanced=True,
        )
        entity_description: str | None = SchemaField(
            default=None,
            description="What each item is, when entity type is 'custom' (2-200 characters)",
            advanced=True,
        )

        # Refresh configuration (for REFRESH behavior)
        refresh_content: bool = SchemaField(
            default=True,
            description="Legacy: only used by refresh monitors, which Exa removed",
            advanced=True,
        )
        refresh_enrichments: bool = SchemaField(
            default=True,
            description="Legacy: only used by refresh monitors, which Exa removed",
            advanced=True,
        )

        # Metadata
        metadata: dict | None = SchemaField(
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
        next_run_at: str | None = SchemaField(
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
                "webset_id": WEBSET_ID,
                "cron_expression": "0 9 * * 1",
                "behavior_type": MonitorBehaviorType.SEARCH,
                "search_query": "AI startups",
                "search_count": 10,
            },
            test_output=[
                ("monitor_id", MONITOR_ID),
                ("webset_id", WEBSET_ID),
                ("status", "enabled"),
                ("behavior_type", "search"),
                ("next_run_at", "2024-01-01T00:00:00Z"),
                ("cron_expression", "0 9 * * 1"),
                ("timezone", "Etc/UTC"),
                ("created_at", "2024-01-01T00:00:00Z"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *args, **kwargs: sample_monitor()},
            effect=BlockEffect.EXTERNAL,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if input_data.behavior_type == MonitorBehaviorType.REFRESH:
            raise ValueError(
                "Exa no longer supports refresh monitors; monitors can only run "
                "searches. Set the behavior type to 'search'."
            )
        payload: dict[str, Any] = {
            "websetId": input_data.webset_id,
            "cadence": {
                "cron": input_data.cron_expression,
                "timezone": input_data.timezone,
            },
            "behavior": {
                "type": "search",
                "config": search_config_payload(
                    query=input_data.search_query,
                    count=input_data.search_count,
                    criteria=input_data.search_criteria,
                    behavior=input_data.search_behavior,
                    entity_type=input_data.entity_type,
                    entity_description=input_data.entity_description,
                ),
            },
        }
        if input_data.metadata:
            payload["metadata"] = input_data.metadata

        monitor = MonitorModel.from_api(
            await self._api(credentials, "POST", "/monitors", json=payload)
        )

        yield "monitor_id", monitor.id
        yield "webset_id", monitor.webset_id
        yield "status", monitor.status
        yield "behavior_type", monitor.behavior_type
        yield "next_run_at", monitor.next_run_at
        yield "cron_expression", monitor.cron_expression
        yield "timezone", monitor.timezone
        yield "created_at", monitor.created_at


def search_config_payload(
    *,
    query: str | None,
    count: int,
    criteria: list[str],
    behavior: Optional["SearchBehavior"],
    entity_type: str | None,
    entity_description: str | None,
) -> dict[str, Any]:
    """A monitor's ``behavior.config``; unset fields reuse the last search's."""
    config: dict[str, Any] = {"count": count}
    optional = {
        "query": query.strip() if query else None,
        "criteria": criteria_payload(criteria),
        "entity": entity_payload(entity_type, entity_description),
        "behavior": behavior.value if behavior else None,
    }
    config.update({k: v for k, v in optional.items() if v})
    return config


class ExaGetMonitorBlock(ExaWebsetsBlockBase):
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
        status: str = SchemaField(description="Current status: enabled or disabled")
        behavior_type: str = SchemaField(description="Type of monitor behavior")
        behavior_config: dict = SchemaField(
            description="Configuration for the monitor behavior"
        )
        cron_expression: str = SchemaField(description="The schedule cron expression")
        timezone: str = SchemaField(description="The timezone for scheduling")
        next_run_at: str | None = SchemaField(
            description="When the monitor will next run"
        )
        last_run: dict | None = SchemaField(
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
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        monitor = MonitorModel.from_api(
            await self._api(
                credentials, "GET", f"/monitors/{segment(input_data.monitor_id)}"
            )
        )

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


class ExaUpdateMonitorBlock(ExaWebsetsBlockBase):
    """Update a monitor's configuration."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The ID of the monitor to update",
            placeholder="monitor-id",
        )
        status: MonitorStatus | None = SchemaField(
            default=None,
            description="Turn the monitor on ('enabled') or off ('disabled'). 'paused' is legacy and is sent as 'disabled'.",
        )
        cron_expression: str | None = SchemaField(
            default=None,
            description="New 5-field cron schedule (at most once per day). Required when changing the timezone.",
        )
        timezone: str | None = SchemaField(
            default=None,
            description="New IANA timezone for the schedule (set together with the cron expression)",
            advanced=True,
        )
        search_query: str | None = SchemaField(
            default=None,
            description="New query for each run (2-10,000 characters). Changing any search setting requires 'search count'.",
            advanced=True,
        )
        search_count: int | None = SchemaField(
            default=None,
            description="New maximum number of items per run. Required to change the monitor's search settings.",
            ge=1,
            advanced=True,
        )
        search_criteria: list[str] = SchemaField(
            default_factory=list,
            description="New criteria for each run (up to 5). Leave empty to keep reusing the last search's criteria.",
            advanced=True,
        )
        search_behavior: SearchBehavior | None = SchemaField(
            default=None,
            description="'append' adds new items; 'override' re-evaluates existing items",
            advanced=True,
        )
        entity_type: str | None = SchemaField(
            default=None,
            description="New kind of item to find: company, person, article, research_paper or custom",
            advanced=True,
        )
        entity_description: str | None = SchemaField(
            default=None,
            description="What each item is, when entity type is 'custom' (2-200 characters)",
            advanced=True,
        )
        metadata: dict | None = SchemaField(
            default=None,
            description="New metadata for the monitor",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(
            description="The unique identifier for the monitor"
        )
        status: str = SchemaField(description="Updated status of the monitor")
        next_run_at: str | None = SchemaField(
            description="When the monitor will next run"
        )
        updated_at: str = SchemaField(description="When the monitor was updated")
        success: str = SchemaField(description="Whether the update was successful")

    def __init__(self):
        super().__init__(
            id="245102c3-6af3-4515-a308-c2210b7939d2",
            description="Update a monitor's status, schedule, search settings or metadata",
            categories={BlockCategory.SEARCH},
            input_schema=ExaUpdateMonitorBlock.Input,
            output_schema=ExaUpdateMonitorBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        monitor = MonitorModel.from_api(
            await self._api(
                credentials,
                "PATCH",
                f"/monitors/{segment(input_data.monitor_id)}",
                json=update_monitor_payload(input_data),
            )
        )

        yield "monitor_id", monitor.id
        yield "status", monitor.status
        yield "next_run_at", monitor.next_run_at
        yield "updated_at", monitor.updated_at
        yield "success", "true"


def update_monitor_payload(data: ExaUpdateMonitorBlock.Input) -> dict[str, Any]:
    """Build the ``PATCH /v0/monitors/{id}`` body from the set inputs."""
    payload: dict[str, Any] = {}
    if data.status is not None:
        payload["status"] = (
            MonitorStatus.DISABLED.value
            if data.status == MonitorStatus.PAUSED
            else data.status.value
        )
    if data.cron_expression or data.timezone:
        if not data.cron_expression:
            raise ValueError(
                "Exa needs the cron expression to change a monitor's timezone; "
                "set both."
            )
        payload["cadence"] = {"cron": data.cron_expression}
        if data.timezone:
            payload["cadence"]["timezone"] = data.timezone
    search_changes = (
        data.search_query,
        data.search_criteria,
        data.search_behavior,
        data.entity_type,
    )
    if data.search_count is not None:
        payload["behavior"] = {
            "type": "search",
            "config": search_config_payload(
                query=data.search_query,
                count=data.search_count,
                criteria=data.search_criteria,
                behavior=data.search_behavior,
                entity_type=data.entity_type,
                entity_description=data.entity_description,
            ),
        }
    elif any(search_changes):
        raise ValueError(
            "Set 'search count' to change a monitor's search settings; Exa "
            "requires it whenever the search is updated."
        )
    if data.metadata is not None:
        payload["metadata"] = data.metadata
    return payload


class ExaDeleteMonitorBlock(ExaWebsetsBlockBase):
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
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        deleted = await self._api(
            credentials, "DELETE", f"/monitors/{segment(input_data.monitor_id)}"
        )

        yield "monitor_id", deleted.get("id", input_data.monitor_id)
        yield "success", "true"


class ExaListMonitorsBlock(ExaWebsetsBlockBase):
    """List all monitors with pagination."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str | None = SchemaField(
            default=None,
            description="Filter monitors by webset ID",
            placeholder="webset-id",
        )
        limit: int = SchemaField(
            default=25,
            description="Number of monitors to return (1-200)",
            ge=1,
            le=200,
        )
        cursor: str | None = SchemaField(
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
        next_cursor: str | None = SchemaField(
            description="Cursor for the next page of results"
        )

    def __init__(self):
        super().__init__(
            id="f06e2b38-5397-4e8f-aa85-491149dd98df",
            description="List all monitors with optional webset filtering",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListMonitorsBlock.Input,
            output_schema=ExaListMonitorsBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        response = await self._api(
            credentials,
            "GET",
            "/monitors",
            params={
                "cursor": input_data.cursor,
                "limit": input_data.limit,
                "websetId": input_data.webset_id,
            },
        )
        monitors = [MonitorModel.from_api(m) for m in response.get("data", [])]

        yield "monitors", [m.model_dump() for m in monitors]
        for monitor in monitors:
            yield "monitor", monitor.model_dump()
        yield "has_more", bool(response.get("hasMore"))
        yield "next_cursor", response.get("nextCursor")
