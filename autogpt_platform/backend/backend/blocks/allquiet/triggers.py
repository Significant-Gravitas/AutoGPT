"""Trigger block fired by an All Quiet outbound webhook."""

from typing import Any

from backend.sdk import (
    Block,
    BlockCategory,
    BlockManualWebhookConfig,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    ProviderName,
    SchemaField,
)

from ._config import allquiet
from ._types import IncidentSeverity, IncidentStatus
from ._webhook import AllQuietWebhookType

# Body template to paste into the All Quiet outbound webhook. It forwards the
# whole incident rather than a flattened subset, so the block can emit status,
# severity and attributes as well as the IDs.
RECOMMENDED_BODY_TEMPLATE = """{
  "id": "{{id}}",
  "title": "{{title}}",
  "eventId": "{{events.[0].id}}",
  "status": "{{events.[0].status}}",
  "severity": "{{events.[0].severity}}",
  "intent": "{{events.[0].modification.intent}}",
  "attributes": [
    {{#each attributes}}
      { "name": "{{this.name}}", "value": "{{this.value}}" },
    {{/each}}
  ]
}"""

EXAMPLE_PAYLOAD = {
    "id": "81cd20be-5837-4dd8-a951-2207a025a231",
    "title": "RAM above 60%",
    "eventId": "b4d2a6a9-be50-45ff-863a-b0ee1a77494e",
    "status": "Open",
    "severity": "Warning",
    "intent": "Investigated",
    "attributes": [
        {"name": "Environment", "value": "prod"},
        {"name": "Alert", "value": "RAM > 60%1"},
    ],
}


class AllQuietIncidentTriggerBlock(Block):
    """Starts a graph whenever All Quiet posts an incident to this webhook.

    Configure an outbound webhook integration in All Quiet pointing at this
    block's webhook URL. The block reads All Quiet's default flattened payload
    (``incidentId``/``incidentTitle``/``incidentProperties``) as well as the
    richer template in ``RECOMMENDED_BODY_TEMPLATE``, and always emits the raw
    body so any custom template stays usable.
    """

    class Input(BlockSchemaInput):
        payload: dict = SchemaField(hidden=True, default_factory=dict)
        signing_secret: str | None = SchemaField(
            title="Signing secret",
            description=(
                "Optional. If All Quiet's outbound webhook has signing enabled, "
                "paste the signing secret here and deliveries with a missing or "
                "bad signature are rejected. Both the All Quiet "
                "(x-aq-signature) and AWS (x-amzn-event-signature) formats are "
                "accepted. Leave empty for unsigned webhooks."
            ),
            default=None,
            secret=True,
        )

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(description="The raw payload All Quiet delivered")
        incident_id: str = SchemaField(description="ID of the incident that fired")
        incident_title: str = SchemaField(description="Title of the incident")
        event_id: str = SchemaField(
            description="ID of the timeline event that triggered this delivery"
        )
        status: IncidentStatus = SchemaField(
            description="Incident status, if the payload template includes it"
        )
        severity: IncidentSeverity = SchemaField(
            description="Incident severity, if the payload template includes it"
        )
        attributes: dict[str, str] = SchemaField(
            description="Incident attributes flattened to a name/value mapping"
        )

    def __init__(self):
        super().__init__(
            id="9c2e7b41-3d68-4a05-8f1c-6b4d0e9a2f37",
            description="Triggers a graph when All Quiet posts an incident to this webhook",
            categories={BlockCategory.INPUT, BlockCategory.DEVELOPER_TOOLS},
            input_schema=AllQuietIncidentTriggerBlock.Input,
            output_schema=AllQuietIncidentTriggerBlock.Output,
            webhook_config=BlockManualWebhookConfig(
                provider=ProviderName(allquiet.name),
                webhook_type=AllQuietWebhookType.INCIDENT,
            ),
            test_input={"payload": EXAMPLE_PAYLOAD},
            test_output=[
                ("payload", EXAMPLE_PAYLOAD),
                ("incident_id", "81cd20be-5837-4dd8-a951-2207a025a231"),
                ("incident_title", "RAM above 60%"),
                ("event_id", "b4d2a6a9-be50-45ff-863a-b0ee1a77494e"),
                ("status", IncidentStatus.OPEN),
                ("severity", IncidentSeverity.WARNING),
                (
                    "attributes",
                    {"Environment": "prod", "Alert": "RAM > 60%1"},
                ),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        yield "payload", payload

        # All Quiet's stock template flattens the incident to incidentId /
        # incidentTitle / incidentProperties; a template that forwards the
        # incident itself uses id / title / attributes. Accept both.
        incident_id = _first_str(payload, "incidentId", "id")
        if incident_id:
            yield "incident_id", incident_id

        incident_title = _first_str(payload, "incidentTitle", "title")
        if incident_title:
            yield "incident_title", incident_title

        event_id = _first_str(payload, "eventId")
        if event_id:
            yield "event_id", event_id

        status = _first_str(payload, "status")
        if status in {member.value for member in IncidentStatus}:
            yield "status", IncidentStatus(status)

        severity = _first_str(payload, "severity")
        if severity in {member.value for member in IncidentSeverity}:
            yield "severity", IncidentSeverity(severity)

        yield "attributes", _attributes(payload)


def _first_str(payload: dict[str, Any], *keys: str) -> str:
    """Return the first key present in the payload as a string."""
    for key in keys:
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _attributes(payload: dict[str, Any]) -> dict[str, str]:
    """Flatten All Quiet's ``[{name, value}]`` attribute lists to a mapping."""
    entries = payload.get("attributes") or payload.get("incidentProperties") or []
    flattened: dict[str, str] = {}
    for entry in entries:
        name = (entry or {}).get("name")
        if name:
            flattened[str(name)] = str((entry or {}).get("value", ""))
    return flattened
