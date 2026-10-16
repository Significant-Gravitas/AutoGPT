"""Tests for dropping the CLI's recorded prompt snapshots from a transcript."""

import pytest

from backend.util import json

from . import transcript
from .transcript import strip_for_upload, validate_transcript


def _message(uuid: str, parent: str | None, role: str) -> dict:
    return {
        "type": role,
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": role, "content": [{"type": "text", "text": uuid}]},
    }


def _attachment(uuid: str, parent: str, kind: str, **fields) -> dict:
    return {
        "type": "attachment",
        "uuid": uuid,
        "parentUuid": parent,
        "attachment": {"type": kind, **fields},
    }


def _jsonl(*entries: dict) -> str:
    return "\n".join(json.dumps(entry) for entry in entries) + "\n"


def _entries(content: str) -> list[dict]:
    return [json.loads(line) for line in content.strip().split("\n")]


# The shape CLI 2.1.284 writes on a session's first turn: one snapshot before
# the first reply (prompt only) and one after it (prompt and tool list).
_SESSION = _jsonl(
    _message("u1", None, "user"),
    _attachment("hook", "u1", "hook_additional_context", content=["<ctx/>"]),
    _attachment("snap1", "hook", "prompt_snapshot", systemPrompt=["PROMPT"]),
    _message("a1", "snap1", "assistant"),
    _attachment("snap2", "a1", "prompt_snapshot", systemPrompt=["PROMPT"], tools=[]),
    _message("u2", "snap2", "user"),
    _attachment("reminder", "u2", "total_tokens_reminder"),
    _message("a2", "reminder", "assistant"),
)


@pytest.fixture
def strip_on(mocker):
    mocker.patch.object(transcript.config, "claude_agent_strip_prompt_snapshots", True)
    mocker.patch.object(transcript.config, "claude_agent_live_system_prompt", True)


@pytest.fixture
def strip_off(mocker):
    mocker.patch.object(transcript.config, "claude_agent_strip_prompt_snapshots", False)


def test_prompt_snapshots_are_removed(strip_on):
    stripped = _entries(strip_for_upload(_SESSION))

    assert [entry["uuid"] for entry in stripped] == [
        "u1",
        "hook",
        "a1",
        "u2",
        "reminder",
        "a2",
    ]


def test_children_of_a_snapshot_are_reparented(strip_on):
    stripped = {entry["uuid"]: entry for entry in _entries(strip_for_upload(_SESSION))}

    assert stripped["a1"]["parentUuid"] == "hook"
    assert stripped["u2"]["parentUuid"] == "a1"


def test_consecutive_snapshots_reparent_to_the_nearest_kept_entry(strip_on):
    content = _jsonl(
        _message("u1", None, "user"),
        _attachment("snap1", "u1", "prompt_snapshot"),
        _attachment("snap2", "snap1", "prompt_snapshot"),
        _message("a1", "snap2", "assistant"),
    )

    stripped = _entries(strip_for_upload(content))

    assert [entry["uuid"] for entry in stripped] == ["u1", "a1"]
    assert stripped[1]["parentUuid"] == "u1"


def test_other_attachments_are_kept_as_written(strip_on):
    """The hook's additional context is the only copy of a turn's injected
    context; it must reach the next resume byte for byte."""
    kept = strip_for_upload(_SESSION).split("\n")

    for line in _SESSION.split("\n"):
        if '"hook_additional_context"' in line or '"total_tokens_reminder"' in line:
            assert line in kept


def test_stripped_transcript_is_still_valid(strip_on):
    assert validate_transcript(strip_for_upload(_SESSION))


def test_switch_off_leaves_the_file_alone(strip_off):
    assert strip_for_upload(_SESSION) == _SESSION


def test_live_prompt_off_keeps_the_snapshots(mocker):
    """A launch without the live prompt reads the recorded one; with none on
    file it would send and re-record its own prompt on every turn."""
    mocker.patch.object(transcript.config, "claude_agent_strip_prompt_snapshots", True)
    mocker.patch.object(transcript.config, "claude_agent_live_system_prompt", False)

    assert strip_for_upload(_SESSION) == _SESSION
