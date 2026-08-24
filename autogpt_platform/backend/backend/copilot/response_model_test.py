"""Tests for SSE serialization of copilot stream events."""

import json

import pytest
from pydantic import ValidationError

from backend.copilot.response_model import (
    ResponseType,
    StreamCompactionProgress,
    StreamModeChanged,
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
