"""Blocks that create and advance All Quiet incidents."""

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
    AllQuietUser,
    Incident,
    IncidentIntent,
    IncidentSeverity,
    IncidentStatus,
)


class AllQuietCreateIncidentBlock(Block):
    """Raise an incident, which routes to whoever is on call for the given teams."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = allquiet.credentials_field(
            description="The All Quiet integration requires an API Key."
        )
        title: str = SchemaField(
            description="Short summary of what is wrong. Shown in every alert.",
            placeholder="Checkout latency above SLO",
        )
        severity: IncidentSeverity = SchemaField(
            description="How urgent the incident is.",
            default=IncidentSeverity.WARNING,
            advanced=False,
        )
        status: IncidentStatus = SchemaField(
            description=(
                "Open pages the on-call responder. Resolved records the incident "
                "without paging anyone."
            ),
            default=IncidentStatus.OPEN,
            advanced=False,
        )
        message: str = SchemaField(
            description="Longer description with context for the responder.",
            default="",
            advanced=False,
        )
        team_ids: list[str] = SchemaField(
            description=(
                "Teams to route the incident to. Leave empty to use the "
                "integration's default routing."
            ),
            default_factory=list,
            advanced=False,
        )
        service_ids: list[str] = SchemaField(
            description="Affected services, used for status pages and uptime.",
            default_factory=list,
            advanced=True,
        )
        user_ids: list[str] = SchemaField(
            description="Users to assign directly, in addition to on-call routing.",
            default_factory=list,
            advanced=True,
        )
        attributes: dict[str, str] = SchemaField(
            description="Extra key/value context, e.g. host or runbook URL.",
            default_factory=dict,
            advanced=True,
        )
        message_is_public: bool = SchemaField(
            description="Show the message on a connected public status page.",
            default=False,
            advanced=True,
        )
        region: AllQuietRegion = region_field()

    class Output(BlockSchemaOutput):
        incident: Incident = SchemaField(description="The incident that was created")
        incident_id: str = SchemaField(
            description="ID of the new incident, for later get/update calls"
        )
        on_call_users: list[AllQuietUser] = SchemaField(
            description=(
                "Users the incident was routed to. Often empty in the create "
                "response because All Quiet resolves routing asynchronously — "
                "read the incident back with Get Incident to see the responders."
            )
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="6b1f0e2a-8c4d-4a17-9f3e-2d5b7c9a1e40",
            description="Creates an incident in All Quiet and pages the on-call responder",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.ISSUE_TRACKING},
            input_schema=AllQuietCreateIncidentBlock.Input,
            output_schema=AllQuietCreateIncidentBlock.Output,
            test_input={
                "title": "Checkout latency above SLO",
                "severity": IncidentSeverity.CRITICAL.value,
                "status": IncidentStatus.OPEN.value,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("incident", TEST_INCIDENT),
                ("incident_id", TEST_INCIDENT.id),
                ("on_call_users", []),
            ],
            test_mock={"create_incident": lambda *args, **kwargs: TEST_INCIDENT},
        )

    @staticmethod
    async def create_incident(
        credentials: APIKeyCredentials, region: AllQuietRegion, input_data: Input
    ) -> Incident:
        client = AllQuietClient(credentials, region)
        return await client.create_incident(
            title=input_data.title,
            status=input_data.status,
            severity=input_data.severity,
            message=input_data.message,
            message_is_public=input_data.message_is_public,
            team_ids=input_data.team_ids,
            service_ids=input_data.service_ids,
            user_ids=input_data.user_ids,
            attributes=input_data.attributes,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        incident = await self.create_incident(
            credentials, input_data.region, input_data
        )
        yield "incident", incident
        yield "incident_id", incident.id
        yield "on_call_users", incident.on_call_users


class AllQuietUpdateIncidentBlock(Block):
    """Investigate, resolve, escalate or comment on an existing incident."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = allquiet.credentials_field(
            description="The All Quiet integration requires an API Key."
        )
        incident_id: str = SchemaField(description="ID of the incident to update")
        intent: IncidentIntent = SchemaField(
            description=(
                "The transition to apply. An incident only accepts the intents "
                "listed in its allowed_intents — e.g. Investigated/Resolved on an "
                "open incident, Unresolved on a resolved one."
            ),
            default=IncidentIntent.INVESTIGATED,
            advanced=False,
        )
        message: str = SchemaField(
            description="Note recorded on the incident timeline with this change.",
            default="",
            advanced=False,
        )
        severity: Optional[IncidentSeverity] = SchemaField(
            description="Optionally change the severity at the same time.",
            default=None,
            advanced=True,
        )
        message_is_public: bool = SchemaField(
            description="Show the message on a connected public status page.",
            default=False,
            advanced=True,
        )
        region: AllQuietRegion = region_field()

    class Output(BlockSchemaOutput):
        incident: Incident = SchemaField(description="The incident after the update")
        allowed_intents: list[str] = SchemaField(
            description=(
                "Transitions the incident accepts after this update, so a graph "
                "can pick its next intent without re-reading the incident"
            )
        )
        status: Optional[IncidentStatus] = SchemaField(
            description="Status after the update, when the incident reports one"
        )
        severity: Optional[IncidentSeverity] = SchemaField(
            description="Severity after the update, when the incident reports one"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="0d7c4a91-5e28-4b6f-8a03-1c9e6f2b4d75",
            description="Investigates, resolves, escalates or comments on an All Quiet incident",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.ISSUE_TRACKING},
            input_schema=AllQuietUpdateIncidentBlock.Input,
            output_schema=AllQuietUpdateIncidentBlock.Output,
            test_input={
                "incident_id": TEST_INCIDENT.id,
                "intent": IncidentIntent.RESOLVED.value,
                "message": "Rolled back the bad deploy",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("incident", TEST_INCIDENT),
                ("allowed_intents", TEST_INCIDENT.allowed_intents),
                ("status", IncidentStatus.OPEN),
                ("severity", IncidentSeverity.CRITICAL),
            ],
            test_mock={"update_incident": lambda *args, **kwargs: TEST_INCIDENT},
        )

    @staticmethod
    async def update_incident(
        credentials: APIKeyCredentials, region: AllQuietRegion, input_data: Input
    ) -> Incident:
        client = AllQuietClient(credentials, region)
        await client.append_intent(
            input_data.incident_id,
            intent=input_data.intent.value,
            message=input_data.message,
            message_is_public=input_data.message_is_public,
            severity=input_data.severity,
        )
        # The patch endpoint does not echo the updated incident, so read it back.
        return await client.get_incident(input_data.incident_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        incident = await self.update_incident(
            credentials, input_data.region, input_data
        )
        yield "incident", incident
        yield "allowed_intents", incident.allowed_intents
        if incident.status:
            yield "status", incident.status
        if incident.severity:
            yield "severity", incident.severity
