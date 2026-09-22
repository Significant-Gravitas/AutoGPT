"""Tests for sdk/langfuse_events.py — Langfuse trace events for SDK turns."""

from unittest.mock import MagicMock, patch

from backend.copilot.sdk.compaction import CompactionStats
from backend.copilot.sdk.langfuse_events import (
    emit_compaction_event,
    emit_turn_usage_event,
)


def _stats(**overrides) -> CompactionStats:
    defaults = {
        "tokens_before": 201403,
        "tokens_after": 95000,
        "messages_before": 91,
        "messages_after": 40,
    }
    defaults.update(overrides)
    return CompactionStats(**defaults)


def _client(mock_get_client) -> MagicMock:
    client = MagicMock()
    mock_get_client.return_value = client
    return client


class TestEmitCompactionEvent:
    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_emits_cycle_with_wire_stats(self, mock_get_client):
        client = _client(mock_get_client)
        emit_compaction_event(path="sdk_internal", stats=_stats())
        client.create_event.assert_called_once()
        kwargs = client.create_event.call_args.kwargs
        assert kwargs["name"] == "copilot-compaction"
        assert kwargs["metadata"]["path"] == "sdk_internal"
        assert kwargs["metadata"]["tokensBefore"] == 201403
        assert kwargs["metadata"]["tokensAfter"] == 95000
        assert kwargs["metadata"]["messagesBefore"] == 91
        assert kwargs["metadata"]["messagesAfter"] == 40
        assert kwargs["metadata"]["dropped"] is False

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_none_stats_still_records_the_cycle(self, mock_get_client):
        client = _client(mock_get_client)
        emit_compaction_event(path="pre_query", stats=None)
        client.create_event.assert_called_once()
        metadata = client.create_event.call_args.kwargs["metadata"]
        assert metadata["path"] == "pre_query"
        assert "tokensBefore" not in metadata
        assert metadata["dropped"] is False

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_dropped_flag_carried(self, mock_get_client):
        client = _client(mock_get_client)
        emit_compaction_event(
            path="pre_query",
            stats=CompactionStats(dropped=True, messages_before=9),
        )
        metadata = client.create_event.call_args.kwargs["metadata"]
        assert metadata["dropped"] is True
        assert metadata["messagesBefore"] == 9

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_emit_failure_is_silent(self, mock_get_client):
        mock_get_client.side_effect = RuntimeError("langfuse unconfigured")
        emit_compaction_event(path="sdk_internal", stats=_stats())

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_after_source_and_schema_version(self, mock_get_client):
        client = _client(mock_get_client)
        emit_compaction_event(
            path="sdk_internal", stats=_stats(), after_source="no_summary_line"
        )
        metadata = client.create_event.call_args.kwargs["metadata"]
        assert metadata["after_source"] == "no_summary_line"
        assert metadata["schema"] == 2

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_after_source_omitted_when_absent(self, mock_get_client):
        client = _client(mock_get_client)
        emit_compaction_event(path="sdk_internal", stats=_stats())
        metadata = client.create_event.call_args.kwargs["metadata"]
        assert "after_source" not in metadata
        assert metadata["schema"] == 2

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_explicit_trace_id_pins_the_event_to_that_trace(self, mock_get_client):
        client = _client(mock_get_client)
        emit_compaction_event(path="sdk_internal", stats=_stats(), trace_id="trace-7")
        kwargs = client.create_event.call_args.kwargs
        assert kwargs["trace_context"] == {"trace_id": "trace-7"}
        assert kwargs["name"] == "copilot-compaction"

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_no_trace_id_falls_back_to_the_ambient_span(self, mock_get_client):
        client = _client(mock_get_client)
        emit_compaction_event(path="sdk_internal", stats=_stats())
        assert "trace_context" not in client.create_event.call_args.kwargs


class TestEmitTurnUsageEvent:
    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_emits_usage_on_trace_context(self, mock_get_client):
        client = _client(mock_get_client)
        emit_turn_usage_event(
            trace_id="trace-1",
            prompt_tokens=33966,
            completion_tokens=1059,
            cache_read_tokens=64512,
            cache_creation_tokens=0,
            cost_usd=None,
            model="gpt-6-astra",
            provider="codex",
        )
        client.create_event.assert_called_once()
        kwargs = client.create_event.call_args.kwargs
        assert kwargs["trace_context"] == {"trace_id": "trace-1"}
        assert kwargs["name"] == "copilot-turn-usage"
        assert kwargs["metadata"]["prompt_tokens"] == 33966
        assert kwargs["metadata"]["completion_tokens"] == 1059
        assert kwargs["metadata"]["cache_read_tokens"] == 64512
        assert kwargs["metadata"]["cache_creation_tokens"] == 0
        assert kwargs["metadata"]["model"] == "gpt-6-astra"
        assert kwargs["metadata"]["provider"] == "codex"
        assert kwargs["metadata"]["schema"] == 2

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_boundary_peak_estimate_key(self, mock_get_client):
        client = _client(mock_get_client)
        emit_turn_usage_event(
            trace_id="trace-1",
            prompt_tokens=1,
            completion_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            cost_usd=None,
            model="gpt-6-astra",
            provider="codex",
            codex_boundary_peak_estimate=244800,
        )
        metadata = client.create_event.call_args.kwargs["metadata"]
        assert metadata["codex_boundary_peak_estimate"] == 244800

        client.reset_mock()
        emit_turn_usage_event(
            trace_id="trace-1",
            prompt_tokens=1,
            completion_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            cost_usd=None,
            model="m",
            provider="anthropic",
        )
        metadata = client.create_event.call_args.kwargs["metadata"]
        assert "codex_boundary_peak_estimate" not in metadata

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_inner_codex_gauge_included_when_provided(self, mock_get_client):
        client = _client(mock_get_client)
        emit_turn_usage_event(
            trace_id="trace-1",
            prompt_tokens=33966,
            completion_tokens=1059,
            cache_read_tokens=64512,
            cache_creation_tokens=0,
            cost_usd=None,
            model="gpt-6-astra",
            provider="codex",
            codex_input_tokens=98578,
            codex_cached_input_tokens=64512,
        )
        metadata = client.create_event.call_args.kwargs["metadata"]
        assert metadata["codex_input_tokens"] == 98578
        assert metadata["codex_cached_input_tokens"] == 64512

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_inner_codex_gauge_omitted_when_absent(self, mock_get_client):
        client = _client(mock_get_client)
        emit_turn_usage_event(
            trace_id="trace-1",
            prompt_tokens=1,
            completion_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            cost_usd=None,
            model="m",
            provider="anthropic",
        )
        metadata = client.create_event.call_args.kwargs["metadata"]
        assert "codex_input_tokens" not in metadata
        assert "codex_cached_input_tokens" not in metadata

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_none_trace_id_skips_emit(self, mock_get_client):
        emit_turn_usage_event(
            trace_id=None,
            prompt_tokens=1,
            completion_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            cost_usd=None,
            model="m",
            provider="codex",
        )
        mock_get_client.assert_not_called()

    @patch("backend.copilot.sdk.langfuse_events.get_client")
    def test_emit_failure_is_silent(self, mock_get_client):
        mock_get_client.side_effect = RuntimeError("langfuse unconfigured")
        emit_turn_usage_event(
            trace_id="trace-1",
            prompt_tokens=1,
            completion_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            cost_usd=None,
            model="m",
            provider="codex",
        )
