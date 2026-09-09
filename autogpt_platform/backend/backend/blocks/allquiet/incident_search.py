"""Blocks that read All Quiet incidents."""

import asyncio
from typing import Optional

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

from ._api import AllQuietClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, allquiet, region_field
from ._testdata import TEST_INCIDENT
from ._types import (
    AllQuietRegion,
    Incident,
    IncidentSeverity,
    IncidentSortBy,
    IncidentStatus,
)


class AllQuietGetIncidentBlock(Block):
    """Fetch one incident by ID, including which transitions it currently allows."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = allquiet.credentials_field(
            description="The All Quiet integration requires an API Key."
        )
        incident_id: str = SchemaField(description="ID of the incident to fetch")
        include_markdown: bool = SchemaField(
            description=(
                "Also fetch an LLM-friendly markdown report of the incident, "
                "including its attributes and full timeline. Costs one extra "
                "API call."
            ),
            default=False,
            advanced=True,
        )
        region: AllQuietRegion = region_field()

    class Output(BlockSchemaOutput):
        incident: Incident = SchemaField(description="The incident")
        status: Optional[IncidentStatus] = SchemaField(
            description="Current status, when the incident reports one"
        )
        severity: Optional[IncidentSeverity] = SchemaField(
            description="Current severity, when the incident reports one"
        )
        allowed_intents: list[str] = SchemaField(
            description="Transitions this incident currently accepts"
        )
        markdown: str = SchemaField(
            description="Markdown report, if include_markdown was set"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="3f8a2c7e-9d14-4e5b-b806-7a2f1c4e9d63",
            description="Fetches a single All Quiet incident by ID",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.ISSUE_TRACKING},
            input_schema=AllQuietGetIncidentBlock.Input,
            output_schema=AllQuietGetIncidentBlock.Output,
            test_input={
                "incident_id": TEST_INCIDENT.id,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("incident", TEST_INCIDENT),
                ("status", IncidentStatus.OPEN),
                ("severity", IncidentSeverity.CRITICAL),
                ("allowed_intents", TEST_INCIDENT.allowed_intents),
            ],
            test_mock={
                "get_incident": lambda *args, **kwargs: (TEST_INCIDENT, ""),
            },
        )

    @staticmethod
    async def get_incident(
        credentials: APIKeyCredentials, region: AllQuietRegion, input_data: Input
    ) -> tuple[Incident, str]:
        client = AllQuietClient(credentials, region)
        if not input_data.include_markdown:
            return await client.get_incident(input_data.incident_id), ""

        # Two independent reads of the same incident; run them concurrently
        # rather than paying both round-trips in series.
        incident, markdown = await asyncio.gather(
            client.get_incident(input_data.incident_id),
            client.get_incident_markdown(input_data.incident_id),
        )
        return incident, markdown

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        incident, markdown = await self.get_incident(
            credentials, input_data.region, input_data
        )
        yield "incident", incident
        if incident.status:
            yield "status", incident.status
        if incident.severity:
            yield "severity", incident.severity
        yield "allowed_intents", incident.allowed_intents
        if markdown:
            yield "markdown", markdown


class AllQuietListIncidentsBlock(Block):
    """Search incidents, e.g. "every unattended critical still open"."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = allquiet.credentials_field(
            description="The All Quiet integration requires an API Key."
        )
        statuses: list[IncidentStatus] = SchemaField(
            description="Only return incidents in these statuses.",
            default_factory=lambda: [IncidentStatus.OPEN],
            advanced=False,
        )
        severities: list[IncidentSeverity] = SchemaField(
            description="Only return incidents at these severities.",
            default_factory=list,
            advanced=False,
        )
        team_ids: list[str] = SchemaField(
            description="Only return incidents routed to these teams.",
            default_factory=list,
            advanced=False,
        )
        search_term: str = SchemaField(
            description="Free-text match against the incident title.",
            default="",
            advanced=False,
        )
        unattended: Optional[bool] = SchemaField(
            description="Set true to return only incidents nobody has picked up.",
            default=None,
            advanced=False,
        )
        limit: int = SchemaField(
            description="Maximum number of incidents to return.",
            default=25,
            ge=1,
            le=100,
            advanced=False,
        )
        offset: int = SchemaField(
            description="Number of incidents to skip, for paging.",
            default=0,
            ge=0,
            advanced=True,
        )
        created_from: str = SchemaField(
            description="Only incidents created at or after this ISO-8601 timestamp.",
            default="",
            advanced=True,
        )
        created_until: str = SchemaField(
            description="Only incidents created at or before this ISO-8601 timestamp.",
            default="",
            advanced=True,
        )
        sort_by: IncidentSortBy = SchemaField(
            description="Field to sort the results by.",
            default=IncidentSortBy.CREATED,
            advanced=True,
        )
        ascending: bool = SchemaField(
            description="Sort oldest first instead of newest first.",
            default=False,
            advanced=True,
        )
        region: AllQuietRegion = region_field()

    class Output(BlockSchemaOutput):
        incidents: list[Incident] = SchemaField(description="All matching incidents")
        incident: Incident = SchemaField(
            description="Each matching incident, emitted one at a time"
        )
        incident_ids: list[str] = SchemaField(
            description="IDs of the matching incidents"
        )
        count: int = SchemaField(description="Number of incidents returned")
        has_more: bool = SchemaField(
            description="Whether more incidents are available beyond this page"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="c4e91b06-7d3a-4f28-9e51-8b0a2c6d3f97",
            description="Searches All Quiet incidents by status, severity, team or text",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.ISSUE_TRACKING},
            input_schema=AllQuietListIncidentsBlock.Input,
            output_schema=AllQuietListIncidentsBlock.Output,
            test_input={
                "statuses": [IncidentStatus.OPEN.value],
                "limit": 25,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("incidents", [TEST_INCIDENT]),
                ("incident", TEST_INCIDENT),
                ("incident_ids", [TEST_INCIDENT.id]),
                ("count", 1),
                ("has_more", False),
            ],
            test_mock={
                "list_incidents": lambda *args, **kwargs: ([TEST_INCIDENT], False),
            },
        )

    @staticmethod
    async def list_incidents(
        credentials: APIKeyCredentials, region: AllQuietRegion, input_data: Input
    ) -> tuple[list[Incident], bool]:
        client = AllQuietClient(credentials, region)
        return await client.list_incidents(
            statuses=input_data.statuses,
            severities=input_data.severities,
            team_ids=input_data.team_ids,
            search_term=input_data.search_term,
            unattended=input_data.unattended,
            created_from=input_data.created_from,
            created_until=input_data.created_until,
            limit=input_data.limit,
            offset=input_data.offset,
            sort_by=input_data.sort_by,
            ascending=input_data.ascending,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        incidents, has_more = await self.list_incidents(
            credentials, input_data.region, input_data
        )
        yield "incidents", incidents
        for incident in incidents:
            yield "incident", incident
        yield "incident_ids", [incident.id for incident in incidents]
        yield "count", len(incidents)
        yield "has_more", has_more
