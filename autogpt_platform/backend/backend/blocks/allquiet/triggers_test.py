"""Tests that the trigger block reads both All Quiet webhook payload shapes."""

import html
import json
import re

from backend.blocks.allquiet._types import IncidentSeverity, IncidentStatus
from backend.blocks.allquiet.triggers import (
    EXAMPLE_PAYLOAD,
    RECOMMENDED_BODY_TEMPLATE,
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


class TestRecommendedTemplateIsValid:
    """The template is copy-paste guidance, so it must render valid JSON."""

    def test_renders_valid_json_for_multiple_attributes(self):
        rendered = _render(
            RECOMMENDED_BODY_TEMPLATE,
            attributes=[
                {"name": "Environment", "value": "prod"},
                {"name": "Alert", "value": "RAM > 60%"},
            ],
        )

        # Values arrive HTML-escaped (the template uses the JSON-safe `{{ }}`
        # form); the block unescapes them on the way in.
        parsed = json.loads(rendered)
        assert parsed["attributes"] == [
            {"name": "Environment", "value": "prod"},
            {"name": "Alert", "value": "RAM &gt; 60%"},
        ]
        assert [html.unescape(a["value"]) for a in parsed["attributes"]] == [
            "prod",
            "RAM > 60%",
        ]

    def test_renders_valid_json_for_one_attribute(self):
        rendered = _render(
            RECOMMENDED_BODY_TEMPLATE, attributes=[{"name": "Env", "value": "prod"}]
        )

        assert json.loads(rendered)["attributes"] == [{"name": "Env", "value": "prod"}]

    def test_renders_valid_json_for_no_attributes(self):
        assert (
            json.loads(_render(RECOMMENDED_BODY_TEMPLATE, attributes=[]))["attributes"]
            == []
        )


def _render(template: str, *, attributes: list[dict]) -> str:
    """Minimal Handlebars stand-in for the subset the template uses.

    Only `{{#each}}` / `{{#unless @last}}` / `{{field}}` are needed, so this
    avoids adding a Handlebars dependency just to assert the template is valid.
    """
    each = re.search(r"\{\{#each attributes\}\}(.*?)\{\{/each\}\}", template, re.DOTALL)
    assert each, "template no longer contains the attributes loop"

    body = each.group(1)
    rendered_items = []
    for index, attribute in enumerate(attributes):
        item = body
        separator = "" if index == len(attributes) - 1 else ","
        item = re.sub(r"\{\{#unless @last\}\},\{\{/unless\}\}", separator, item)
        # Handlebars HTML-escapes {{ }} values; mirror that here.
        item = item.replace("{{this.name}}", html.escape(attribute["name"], quote=True))
        item = item.replace(
            "{{this.value}}", html.escape(attribute["value"], quote=True)
        )
        rendered_items.append(item.strip())

    out = template[: each.start()] + "".join(rendered_items) + template[each.end() :]
    # Fill the remaining scalar placeholders with something JSON-safe.
    return re.sub(r"\{\{+[^}]+\}\}+", "x", out)


class TestHtmlEscaping:
    """All Quiet's stock template (and its "test alert" button) HTML-escapes values.

    Handlebars escapes `{{ }}` by default, so `RAM > 60%` arrives as
    `RAM &gt; 60%`. Captured from a real test-alert delivery.
    """

    async def test_unescapes_attribute_values_from_the_test_alert(self):
        out = await _outputs(
            {
                "eventId": "b4d2a6a9-be50-45ff-863a-b0ee1a77494e",
                "incidentId": "81cd20be-5837-4dd8-a951-2207a025a231",
                "incidentTitle": "",
                "incidentProperties": [
                    {"name": "Environment", "value": "prod"},
                    {"name": "Alert", "value": "RAM &gt; 60%1"},
                ],
            }
        )

        assert out["attributes"] == {
            "Environment": "prod",
            "Alert": "RAM > 60%1",
        }

    async def test_unescapes_the_incident_title(self):
        out = await _outputs({"incidentTitle": "Disk &gt; 90% &amp; climbing"})

        assert out["incident_title"] == "Disk > 90% & climbing"

    async def test_omits_an_empty_title(self):
        # The test alert sends incidentTitle: "" — emitting an empty string
        # would look like a real (blank) title downstream.
        out = await _outputs({"incidentId": "x", "incidentTitle": ""})

        assert "incident_title" not in out

    async def test_round_trips_a_literal_entity(self):
        # A value containing a literal "&gt;" is double-escaped by Handlebars,
        # so unescaping recovers exactly what was typed.
        out = await _outputs({"attributes": [{"name": "n", "value": "&amp;gt;"}]})

        assert out["attributes"]["n"] == "&gt;"

    async def test_unescapes_attribute_names_too(self):
        out = await _outputs({"attributes": [{"name": "A &amp; B", "value": "v"}]})

        assert out["attributes"] == {"A & B": "v"}


class TestRecommendedTemplateEscapes:
    def test_uses_no_triple_stash(self):
        # Triple-stash emits raw values, so an alert title containing a double
        # quote would terminate the JSON string early and break the body. The
        # escaping form keeps it valid; the block unescapes on the way in.
        assert "{{{" not in RECOMMENDED_BODY_TEMPLATE

    def test_a_value_containing_a_quote_still_renders_valid_json(self):
        rendered = _render(
            RECOMMENDED_BODY_TEMPLATE,
            attributes=[{"name": "Alert", "value": 'unterminated "group"'}],
        )

        # Valid JSON, and the escaped value round-trips through the block.
        parsed = json.loads(rendered)
        assert parsed["attributes"][0]["value"] == "unterminated &quot;group&quot;"
        assert html.unescape(parsed["attributes"][0]["value"]) == 'unterminated "group"'
