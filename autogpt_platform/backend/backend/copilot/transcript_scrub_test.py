"""``strip_for_upload`` and the CLI's mention attachments: which attachment
types it drops, which it keeps, and that no kept entry of a compacted session
names a dropped one afterwards."""

from __future__ import annotations

from typing import Any

import pytest

from backend.util import json

from .transcript import strip_for_upload

# Attachment types the CLI writes for what the prompt text names; each must be
# dropped. Listed here rather than read from MENTION_ATTACHMENT_TYPES, so a type
# missing from that set fails its own case.
_MENTION_ATTACHMENT_TYPES = [
    "file",
    "already_read_file",
    "compact_file_reference",
    "pdf_reference",
    "directory",
    "agent_mention",
    "mcp_resource",
    "peer_mention",
    "at_mention_reference",
    "edited_text_file",
    "edited_image_file",
]

# Attachment types a resumed session needs; none may be dropped.
_KEPT_ATTACHMENT_TYPES = [
    "hook_additional_context",
    "prompt_snapshot",
    "environment",
    "model",
    "date",
    "session_context",
    "agent_listing_delta",
    "skill_listing",
    "total_tokens_reminder",
    "ultrathink_effort",
]

# The lines of a session CLI 2.1.284 wrote for a turn that named three files, a
# directory and an agent, a second turn that named a file again and then
# compacted, and a third turn: (label, type, attachment type or subtype,
# parent label). A "last-prompt" line's last field is its leaf label.
_COMPACTED_SESSION: list[tuple[str, str, str, str | None]] = [
    ("q0", "queue-operation", "", None),
    ("u2", "user", "", None),
    ("a3", "attachment", "file", "u2"),
    ("a4", "attachment", "directory", "a3"),
    ("a5", "attachment", "file", "a4"),
    ("a6", "attachment", "file", "a5"),
    ("a7", "attachment", "agent_mention", "a6"),
    ("a8", "attachment", "environment", "a7"),
    ("a9", "attachment", "model", "a8"),
    ("a10", "attachment", "ultrathink_effort", "a9"),
    ("a11", "attachment", "agent_listing_delta", "a10"),
    ("a12", "attachment", "total_tokens_reminder", "a11"),
    ("l13", "atis-latch", "", None),
    ("a14", "attachment", "session_context", "a12"),
    ("a15", "attachment", "date", "a14"),
    ("a16", "attachment", "prompt_snapshot", "a15"),
    ("r17", "assistant", "", "a16"),
    ("a18", "attachment", "prompt_snapshot", "r17"),
    ("p19", "last-prompt", "", "a18"),
    ("c21", "cost-state", "", None),
    ("p22", "last-prompt", "", "a29"),
    ("m23", "mode", "", None),
    ("u27", "user", "", "a18"),
    ("a28", "attachment", "file", "u27"),
    ("a29", "attachment", "total_tokens_reminder", "a28"),
    ("s30", "system", "compact_boundary", None),
    ("a31", "attachment", "session_context", "s30"),
    ("a32", "attachment", "date", "a31"),
    ("u33", "user", "compact_summary", "a32"),
    ("a34", "attachment", "file", "u33"),
    ("a35", "attachment", "agent_listing_delta", "a34"),
    ("a36", "attachment", "environment", "a35"),
    ("a37", "attachment", "model", "a36"),
    ("a38", "attachment", "prompt_snapshot", "a37"),
    ("r39", "assistant", "", "a38"),
    ("a40", "attachment", "prompt_snapshot", "r39"),
    ("p41", "last-prompt", "", "a40"),
    ("u47", "user", "", "a40"),
    ("a48", "attachment", "total_tokens_reminder", "u47"),
    ("r49", "assistant", "", "a48"),
    ("p50", "last-prompt", "", "r49"),
]
# The compact boundary's metadata, as CLI 2.1.284 wrote it for that session.
_PRESERVED = {
    "preservedSegment": {"headUuid": "u27", "anchorUuid": "u33", "tailUuid": "a29"},
    "preservedMessages": {
        "anchorUuid": "u33",
        "uuids": ["u27", "a28", "a29"],
        "allUuids": ["u27", "a28", "a29"],
    },
}
_LOGICAL_PARENT = "a29"


@pytest.mark.parametrize("attachment_type", _MENTION_ATTACHMENT_TYPES)
def test_each_mention_attachment_type_is_left_out(attachment_type: str):
    content = _jsonl(
        _user("u1", None),
        _attachment("m2", "u1", attachment_type),
        _attachment("k3", "m2", "date"),
    )

    entries = _entries(strip_for_upload(content))

    assert [entry["uuid"] for entry in entries] == [_uid("u1"), _uid("k3")]
    assert entries[1]["parentUuid"] == _uid("u1")


def test_api_request_blobs_are_left_out():
    """Newer CLIs keep a copy of each request as sent, attachments included."""
    blob = {
        "type": "api-request-blob",
        "hash": "h1",
        "sessionId": "s1",
        "message": {"role": "user", "content": "an attached file's words"},
    }
    content = _jsonl(_user("u1", None), blob, _assistant("r2", "u1"))

    entries = _entries(strip_for_upload(content))

    assert [entry["type"] for entry in entries] == ["user", "assistant"]


@pytest.mark.parametrize(
    "entry_type",
    [
        "api-request-blob",
        "progress",
        "file-history-snapshot",
        "queue-operation",
        "pr-link",
    ],
)
def test_only_a_summary_is_kept_for_being_a_compact_summary(entry_type: str):
    """Any other strippable entry is left out even when marked as one."""
    marked = {
        "type": entry_type,
        "isCompactSummary": True,
        "message": {"role": "user", "content": "an attached file's words"},
    }
    summary = {
        "type": "summary",
        "isCompactSummary": True,
        "message": {"role": "user", "content": "compacted context"},
    }
    content = _jsonl(_user("u1", None), marked, summary, _assistant("r2", "u1"))

    entries = _entries(strip_for_upload(content))

    assert [entry["type"] for entry in entries] == ["user", "summary", "assistant"]


@pytest.mark.parametrize(
    "unreadable",
    [
        '{"type":"attachment","uuid":"m2","attachment":{"type":"file","con',
        '["type", "attachment"]',
        '"attachment"',
        "null",
    ],
    ids=["truncated", "array", "string", "null"],
)
def test_a_line_that_is_not_a_json_object_stops_the_strip(unreadable: str):
    """It could hold an entry that must be dropped, so nothing is returned."""
    content = _jsonl(_user("u1", None), _assistant("r2", "u1")) + unreadable + "\n"

    with pytest.raises(ValueError):
        strip_for_upload(content)


def test_blank_lines_do_not_stop_the_strip():
    content = _jsonl(_user("u1", None)) + "\n" + _jsonl(_assistant("r2", "u1"))

    assert strip_for_upload(content) == content


@pytest.mark.parametrize("attachment_type", _KEPT_ATTACHMENT_TYPES)
def test_attachment_types_a_resumed_session_needs_are_kept(attachment_type: str):
    content = _jsonl(
        _user("u1", None),
        _attachment("k2", "u1", attachment_type),
        _assistant("r3", "k2"),
    )

    assert strip_for_upload(content) == content


def test_no_kept_entry_of_a_compacted_session_names_a_dropped_one():
    before = _compacted_session(_PRESERVED)

    after = _entries(strip_for_upload(_jsonl(*before)))

    assert _dangling_references(before, after) == []
    boundary = next(entry for entry in after if entry["type"] == "system")
    preserved = boundary["compactMetadata"]["preservedMessages"]
    assert preserved["uuids"] == preserved["allUuids"] == [_uid("u27"), _uid("a29")]
    assert _walk_preserved_segment(after, boundary) == preserved["uuids"]


@pytest.mark.parametrize(
    ("segment", "kept_uuids"),
    [
        ({"headUuid": "a28", "tailUuid": "a29"}, ["a29"]),
        ({"headUuid": "u27", "tailUuid": "a28"}, ["u27"]),
    ],
    ids=["dropped head", "dropped tail"],
)
def test_a_preserved_segment_ending_on_a_dropped_entry_is_moved_inside_it(
    segment: dict[str, str], kept_uuids: list[str]
):
    before = _compacted_session({"preservedSegment": {**segment, "anchorUuid": "u33"}})

    after = _entries(strip_for_upload(_jsonl(*before)))

    assert _dangling_references(before, after) == []
    boundary = next(entry for entry in after if entry["type"] == "system")
    walked = _walk_preserved_segment(after, boundary)
    assert walked == [_uid(label) for label in kept_uuids]


def test_a_preserved_segment_with_nothing_kept_is_removed():
    before = _compacted_session(
        {
            "preservedSegment": {
                "headUuid": "a28",
                "anchorUuid": "u33",
                "tailUuid": "a28",
            }
        }
    )

    after = _entries(strip_for_upload(_jsonl(*before)))

    assert _dangling_references(before, after) == []
    boundary = next(entry for entry in after if entry["type"] == "system")
    assert "preservedSegment" not in boundary["compactMetadata"]


def _compacted_session(preserved: dict[str, dict[str, Any]]) -> list[dict]:
    entries = []
    for label, kind, detail, link in _COMPACTED_SESSION:
        if kind == "user":
            entry = _user(label, link)
            if detail == "compact_summary":
                entry["isCompactSummary"] = True
        elif kind == "assistant":
            entry = _assistant(label, link)
        elif kind == "attachment":
            entry = _attachment(label, link, detail)
        elif kind == "system":
            entry = _boundary(label, preserved)
        elif kind == "last-prompt":
            entry = {"type": kind, "leafUuid": _uid(link)}
        else:
            entry = {"type": kind}
        entries.append(entry)
    return entries


def _boundary(label: str, preserved: dict[str, dict[str, Any]]) -> dict:
    metadata: dict[str, Any] = {"trigger": "auto", "preTokens": 190097}
    for section, fields in preserved.items():
        metadata[section] = {
            key: (
                [_uid(item) for item in value]
                if isinstance(value, list)
                else _uid(value)
            )
            for key, value in fields.items()
        }
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": _uid(label),
        "parentUuid": None,
        "logicalParentUuid": _uid(_LOGICAL_PARENT),
        "compactMetadata": metadata,
    }


def _user(label: str, parent: str | None) -> dict:
    return {
        "type": "user",
        "uuid": _uid(label),
        "parentUuid": _uid(parent),
        "message": {"role": "user", "content": f"text {label}"},
    }


def _assistant(label: str, parent: str | None) -> dict:
    return {
        "type": "assistant",
        "uuid": _uid(label),
        "parentUuid": _uid(parent),
        "message": {"role": "assistant", "id": f"msg_{label}", "content": "ok"},
    }


def _attachment(label: str, parent: str | None, attachment_type: str) -> dict:
    return {
        "type": "attachment",
        "uuid": _uid(label),
        "parentUuid": _uid(parent),
        "attachment": {"type": attachment_type},
    }


def _uid(label: str | None) -> str | None:
    """A uuid-shaped id for *label*, so no other string in a line equals it."""
    if label is None:
        return None
    return f"00000000-0000-4000-8000-{int(label[1:]):012d}"


def _dangling_references(before: list[dict], after: list[dict]) -> list[str]:
    """Every place in a kept entry that names an entry which is not kept,
    found by walking each kept entry recursively."""
    known = {entry["uuid"] for entry in before if "uuid" in entry}
    kept = {entry["uuid"] for entry in after if "uuid" in entry}
    found: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str) and value in known and value not in kept:
            found.append(path)
        elif isinstance(value, str) and _names_an_entry(path) and value not in kept:
            found.append(path)

    for index, entry in enumerate(after):
        for key, value in entry.items():
            if key != "uuid":
                walk(value, f"[{index}].{key}")
    return found


def _names_an_entry(path: str) -> bool:
    field = path.rsplit(".", 1)[-1].split("[", 1)[0]
    return field.endswith(("Uuid", "UUID", "uuids", "Uuids"))


def _walk_preserved_segment(entries: list[dict], boundary: dict) -> list[str]:
    """The ids from the segment's head to its tail along ``parentUuid``, the way
    the CLI walks them when it relinks a compacted session."""
    segment = boundary["compactMetadata"]["preservedSegment"]
    by_uuid = {entry["uuid"]: entry for entry in entries if "uuid" in entry}
    walked: list[str] = []
    current = by_uuid.get(segment["tailUuid"])
    while current is not None and current["uuid"] not in walked:
        walked.append(current["uuid"])
        if current["uuid"] == segment["headUuid"]:
            return walked[::-1]
        current = by_uuid.get(current.get("parentUuid") or "")
    raise AssertionError("the preserved segment's tail does not lead to its head")


def _jsonl(*entries: dict) -> str:
    return "\n".join(json.dumps(entry) for entry in entries) + "\n"


def _entries(content: str) -> list[dict]:
    return [json.loads(line) for line in content.splitlines()]
