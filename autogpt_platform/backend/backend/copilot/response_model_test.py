"""Tests for SSE serialization of copilot stream events."""

import json

from backend.copilot.response_model import StreamModeChanged


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
