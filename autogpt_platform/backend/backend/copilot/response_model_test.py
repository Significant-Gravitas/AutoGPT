"""Tests for SSE serialization of copilot stream events."""

import json

from backend.copilot.response_model import (
    MAX_SUGGESTION_LENGTH,
    MAX_SUGGESTIONS,
    StreamModeChanged,
    StreamSuggestions,
)


def test_mode_changed_serializes_as_ai_sdk_data_part():
    """The frontend reads ``dataPart.data.mode`` — mode must be nested under
    ``data``, not serialized as a top-level sibling of ``type``."""
    sse = StreamModeChanged(mode="extended_thinking").to_sse()
    assert sse.startswith("data: ")
    payload = json.loads(sse[len("data: ") :])
    assert payload == {
        "type": "data-mode-changed",
        "data": {"mode": "extended_thinking"},
    }


def _payload(event: StreamSuggestions) -> dict:
    sse = event.to_sse()
    assert sse.startswith("data: ")
    return json.loads(sse[len("data: ") :])


def test_suggestions_serialize_as_ai_sdk_data_part():
    """Chips must arrive nested under ``data`` so the AI SDK surfaces them
    as a ``data-suggestions`` part on ``message.parts``."""
    event = StreamSuggestions(
        suggestions=["Email the report", "Post on r/SaaS", "Fix the criticals"]
    )
    assert _payload(event) == {
        "type": "data-suggestions",
        "data": {
            "suggestions": [
                "Email the report",
                "Post on r/SaaS",
                "Fix the criticals",
            ]
        },
    }


def test_suggestions_absent_serializes_as_empty_list():
    """A turn with nothing worth suggesting still round-trips — the client
    renders no chips rather than crashing on a missing key."""
    assert _payload(StreamSuggestions()) == {
        "type": "data-suggestions",
        "data": {"suggestions": []},
    }


def test_suggestions_beyond_the_cap_are_truncated():
    """The model is told "up to 3"; a fourth is dropped at the model layer so
    the cap holds for replayed events too, not just freshly published ones."""
    event = StreamSuggestions(
        suggestions=["One", "Two", "Three", "Four", "Five"],
    )
    assert event.suggestions == ["One", "Two", "Three"]
    assert len(_payload(event)["data"]["suggestions"]) == MAX_SUGGESTIONS


def test_suggestions_drop_blanks_and_duplicates_and_clamp_length():
    event = StreamSuggestions(
        suggestions=[
            "  Email   the report  ",
            "",
            "   ",
            "email the report",
            "x" * (MAX_SUGGESTION_LENGTH + 20),
        ]
    )
    assert event.suggestions == [
        "Email the report",
        "x" * MAX_SUGGESTION_LENGTH,
    ]


def test_suggestions_survive_redis_round_trip():
    """Replayed chunks are rebuilt with ``model_validate`` on stream resume;
    normalisation must apply there too."""
    raw = StreamSuggestions(
        suggestions=["A", "B", "C", "D"]
    ).model_dump_json()
    restored = StreamSuggestions.model_validate_json(raw)
    assert restored.suggestions == ["A", "B", "C"]
    assert restored.type.value == "data-suggestions"
