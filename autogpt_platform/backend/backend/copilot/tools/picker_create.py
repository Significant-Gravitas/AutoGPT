"""Name the capability that creates what a picker card would pick, so the
model makes a new spreadsheet or document itself instead of asking the user to."""

from typing import Any

from backend.blocks import get_block
from backend.blocks._base import AnyBlockSchema, BlockEffect
from backend.copilot.capabilities.block_meta import get_block_provider
from backend.copilot.capabilities.registry import get_registry
from backend.copilot.context import get_current_permissions
from backend.copilot.gate.effects import block_effect

from .helpers import is_picker_field
from .models import SetupRequirementsResponse

_ONLY_READS = {BlockEffect.NONE, BlockEffect.READ}


def create_hint(
    block: AnyBlockSchema, card: SetupRequirementsResponse, inputs: dict[str, Any]
) -> str | None:
    """A line per unpicked picker naming what creates a new one, or None when
    the card is a sign-in card, the block only reads, or nothing creates it."""
    readiness = card.setup_info.user_readiness
    if not readiness.has_all_credentials or block_effect(block, inputs) in _ONLY_READS:
        return None
    lines = []
    for field in card.setup_info.requirements.get("inputs", []):
        if field.get("value") not in (None, ""):
            continue
        name = str(field["name"])
        creators = ", ".join(
            f"`block:{c.id}` ({c.name})" for c in creators_of(block, name)
        )
        if creators:
            title = str(field.get("title") or name).lower()
            lines.append(
                f"If the user wants a new {title} rather than one they have, "
                f"don't ask them to make one: create it with {creators} and "
                f"pass its `{name}` output as `{name}`."
            )
    return " ".join(lines) or None


def creators_of(block: AnyBlockSchema, field_name: str) -> list[AnyBlockSchema]:
    """Blocks of the same provider that pick nothing themselves and output a
    value of the picker's own object type under the picker's name."""
    properties = block.input_schema.jsonschema().get("properties", {})
    wanted = _object_fields(properties.get(field_name))
    if wanted is None:
        # A plain string fits any string field, so it is no evidence the
        # output is the thing this picker picks (an Airtable base id).
        return []
    provider = get_block_provider(block)
    permissions = get_current_permissions()
    found = []
    for entry in get_registry().entries:
        if entry.kind != "block" or entry.context == "graph":
            continue
        if provider not in entry.tags:
            continue  # an entry's tags carry its provider; checked before building it
        candidate = get_block(entry.implementations[0].ref)
        if candidate is None:
            continue
        if permissions is not None and not permissions.is_block_allowed(
            candidate.id, candidate.name
        ):
            continue
        candidate_inputs = candidate.input_schema.jsonschema().get("properties", {})
        if any(is_picker_field(p) for p in candidate_inputs.values()):
            continue
        outputs = candidate.output_schema.jsonschema().get("properties", {})
        if _object_fields(outputs.get(field_name)) == wanted:
            found.append(candidate)
    return found


def _object_fields(schema: Any) -> frozenset[str] | None:
    if not isinstance(schema, dict):
        return None
    for variant in (schema, *schema.get("anyOf", [])):
        if isinstance(variant, dict) and variant.get("properties"):
            return frozenset(variant["properties"])
    return None
