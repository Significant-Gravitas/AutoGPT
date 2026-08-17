"""Tests that the trigger block reads both All Quiet webhook payload shapes."""

from backend.blocks.allquiet._types import IncidentSeverity, IncidentStatus
from backend.blocks.allquiet.triggers import (
    EXAMPLE_PAYLOAD,
    AllQuietIncidentTriggerBlock,
)

# All Quiet's stock outbound-webhook body template renders this shape.
DEFAULT_TEMPLATE_PAYLOAD = {
    "eventId": "b4d2a6a9-be50-45ff-863a-b0ee1a77494e",
    "incidentId": "81cd20be-5837-4dd8-a951-2207a025a231",
    "incidentTitle": "RAM above 60%",
    "incidentProperties": [
        {"name": "Environment", "value": "prod"},
        {"name": "Alert", "value": "RAM > 60%1"},
    ],
}


async def _outputs(payload: dict) -> dict:
    block = AllQuietIncidentTriggerBlock()
    collected: dict = {}
    async for name, value in block.run(block.input_schema(payload=payload)):
        collected[name] = value
    return collected


class TestDefaultTemplate:
    async def test_reads_the_flattened_incident_fields(self):
        out = await _outputs(DEFAULT_TEMPLATE_PAYLOAD)

        assert out["incident_id"] == "81cd20be-5837-4dd8-a951-2207a025a231"
        assert out["incident_title"] == "RAM above 60%"
        assert out["event_id"] == "b4d2a6a9-be50-45ff-863a-b0ee1a77494e"

    async def test_flattens_incident_properties(self):
        out = await _outputs(DEFAULT_TEMPLATE_PAYLOAD)

        assert out["attributes"] == {"Environment": "prod", "Alert": "RAM > 60%1"}

    async def test_omits_status_and_severity_the_template_does_not_carry(self):
        out = await _outputs(DEFAULT_TEMPLATE_PAYLOAD)

        assert "status" not in out
        assert "severity" not in out

    async def test_always_passes_the_raw_payload_through(self):
        out = await _outputs(DEFAULT_TEMPLATE_PAYLOAD)

        assert out["payload"] == DEFAULT_TEMPLATE_PAYLOAD


class TestRecommendedTemplate:
    async def test_reads_status_and_severity(self):
        out = await _outputs(EXAMPLE_PAYLOAD)

        assert out["status"] == IncidentStatus.OPEN
        assert out["severity"] == IncidentSeverity.WARNING

    async def test_reads_the_unprefixed_id_and_title(self):
        out = await _outputs(EXAMPLE_PAYLOAD)

        assert out["incident_id"] == "81cd20be-5837-4dd8-a951-2207a025a231"
        assert out["incident_title"] == "RAM above 60%"


class TestUnexpectedPayloads:
    async def test_survives_an_empty_payload(self):
        out = await _outputs({})

        assert out["payload"] == {}
        assert out["attributes"] == {}
        assert "incident_id" not in out

    async def test_ignores_a_status_value_it_does_not_recognise(self):
        # A custom template could send anything; never emit a bogus enum.
        out = await _outputs({"id": "x", "status": "Snoozed", "severity": "Fatal"})

        assert "status" not in out
        assert "severity" not in out
        assert out["incident_id"] == "x"

    async def test_skips_attribute_entries_with_no_name(self):
        out = await _outputs({"attributes": [{"value": "orphan"}, {"name": "ok"}]})

        assert out["attributes"] == {"ok": ""}
