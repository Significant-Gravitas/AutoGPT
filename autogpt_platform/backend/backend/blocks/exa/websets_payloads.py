"""
Builders for the request bodies the Websets API shares across endpoints.

Blocks take flat, builder-friendly inputs (parallel lists of IDs and types);
these functions turn them into the spec's nested shapes and enforce the
spec's limits up front, so a bad value fails with a clear message instead of
a bare 400 from Exa.
"""

from typing import Any

MAX_CRITERIA = 5
MAX_ENRICHMENT_OPTIONS = 150
SCOPE_RELATIONSHIP_LIMIT_MAX = 10
CUSTOM_ENTITY_DESCRIPTION_RANGE = (2, 200)
SOURCE_TYPES = ("import", "webset")


def entity_payload(
    entity_type: str | None, description: str | None
) -> dict[str, Any] | None:
    """``{"type": ...}`` for a fixed entity, ``None`` to let Exa detect it."""
    if not entity_type or entity_type == "auto":
        return None
    if entity_type != "custom":
        return {"type": entity_type}
    low, high = CUSTOM_ENTITY_DESCRIPTION_RANGE
    if not description or not low <= len(description.strip()) <= high:
        raise ValueError(
            f"A custom entity needs a description of {low}-{high} characters "
            "saying what each result is (for example 'Open-source LLM repos')."
        )
    return {"type": "custom", "description": description.strip()}


def criteria_payload(criteria: list[str]) -> list[dict[str, str]] | None:
    """Criteria objects, or ``None`` so Exa generates them from the query."""
    descriptions = [c.strip() for c in criteria if c and c.strip()]
    if not descriptions:
        return None
    if len(descriptions) > MAX_CRITERIA:
        raise ValueError(
            f"Exa accepts at most {MAX_CRITERIA} criteria per search; "
            f"got {len(descriptions)}."
        )
    return [{"description": d} for d in descriptions]


def sources_payload(ids: list[str], types: list[str]) -> list[dict[str, str]]:
    """``[{"source", "id"}]`` pairs; a missing type defaults to ``import``."""
    return [{"source": _source_type(types, idx), "id": sid} for idx, sid in _ids(ids)]


def import_payload(
    ids: list[str], types: list[str], evaluate: list[bool]
) -> list[dict[str, Any]]:
    """Webset ``import`` entries, with ``evaluate`` where one was given."""
    entries: list[dict[str, Any]] = []
    for idx, sid in _ids(ids):
        entry: dict[str, Any] = {"source": _source_type(types, idx), "id": sid}
        if idx < len(evaluate):
            entry["evaluate"] = evaluate[idx]
        entries.append(entry)
    return entries


def scope_payload(
    ids: list[str],
    types: list[str],
    relationships: list[str],
    limits: list[int],
) -> list[dict[str, Any]]:
    """Search ``scope`` entries; a relationship turns the entry into a hop search."""
    entries: list[dict[str, Any]] = []
    for idx, sid in _ids(ids):
        entry: dict[str, Any] = {"source": _source_type(types, idx), "id": sid}
        definition = relationships[idx].strip() if idx < len(relationships) else ""
        if definition:
            limit = limits[idx] if idx < len(limits) else None
            entry["relationship"] = _relationship(definition, limit)
        entries.append(entry)
    return entries


def options_payload(labels: list[str]) -> list[dict[str, str]]:
    """Option objects for an ``options``-format enrichment (1-150 labels)."""
    if len(labels) > MAX_ENRICHMENT_OPTIONS:
        raise ValueError(
            f"Exa accepts at most {MAX_ENRICHMENT_OPTIONS} options per "
            f"enrichment; got {len(labels)}."
        )
    return [{"label": label} for label in labels]


def enrichment_payload(
    description: str,
    format: str | None,
    options: list[str],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """One ``CreateEnrichmentParameters`` object."""
    payload: dict[str, Any] = {"description": description}
    if format:
        payload["format"] = format
    if options:
        payload["options"] = options_payload(options)
    if metadata:
        payload["metadata"] = metadata
    return payload


def _ids(ids: list[str]) -> list[tuple[int, str]]:
    return [(idx, sid) for idx, sid in enumerate(ids) if sid]


def _source_type(types: list[str], idx: int) -> str:
    source = types[idx] if idx < len(types) and types[idx] else "import"
    if source not in SOURCE_TYPES:
        raise ValueError(f"Source type must be 'import' or 'webset'; got '{source}'.")
    return source


def _relationship(definition: str, limit: int | None) -> dict[str, Any]:
    if limit is None or not 1 <= limit <= SCOPE_RELATIONSHIP_LIMIT_MAX:
        raise ValueError(
            "A scope relationship needs a limit from 1 to "
            f"{SCOPE_RELATIONSHIP_LIMIT_MAX}: how many related entities to find "
            f"per source item (got {limit})."
        )
    return {"definition": definition, "limit": limit}
