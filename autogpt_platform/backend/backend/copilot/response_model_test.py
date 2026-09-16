"""Tests for SSE serialization of copilot stream events."""

import json

import pytest
from pydantic import ValidationError

from backend.copilot.response_model import (
    ResponseType,
    StreamCompactionProgress,
    StreamToolDisplayAvailable,
    ToolDisplayData,
)


def test_tool_display_serializes_as_persistent_ai_sdk_data_part():
    event = StreamToolDisplayAvailable(
        id="call-1",
        data=ToolDisplayData(toolCallId="call-1", displayName='Résumé "Daily"'),
    )
    assert json.loads(event.to_sse().removeprefix("data: ")) == {
        "type": "data-tool-display",
        "id": "call-1",
        "data": {"toolCallId": "call-1", "displayName": 'Résumé "Daily"'},
    }


class TestStreamCompactionProgress:
    def test_type_is_data_compaction(self):
        evt = StreamCompactionProgress(phase="summarizing")
        assert evt.type == ResponseType.COMPACTION
        assert ResponseType.COMPACTION.value == "data-compaction"

    def test_to_sse_wraps_fields_in_data_envelope(self):
        evt = StreamCompactionProgress(
            phase="rebuilding",
            tokensBefore=128_000,
            tokensAfter=31_000,
            messagesBefore=412,
            messagesAfter=38,
        )
        line = evt.to_sse()
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        payload = json.loads(line[len("data: ") : -2])
        assert payload["type"] == "data-compaction"
        assert payload["data"] == {
            "phase": "rebuilding",
            "tokensBefore": 128_000,
            "tokensAfter": 31_000,
            "messagesBefore": 412,
            "messagesAfter": 38,
        }

    def test_to_sse_omits_unknown_stats(self):
        evt = StreamCompactionProgress(phase="summarizing")
        payload = json.loads(evt.to_sse()[len("data: ") : -2])
        assert payload["data"] == {"phase": "summarizing"}

    def test_phase_is_constrained_to_the_emitted_stages(self):
        """The wire contract is typed, not prose.

        ``done`` was documented for a long time but never emitted; the
        frontend indexes a curve table by this value, so an unmodelled
        phase reaching it is a client-side crash rather than a no-op.
        """
        for phase in ("summarizing", "rebuilding"):
            assert StreamCompactionProgress(phase=phase).phase == phase

        with pytest.raises(ValidationError):
            StreamCompactionProgress(phase="done")
