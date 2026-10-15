import asyncio
import json
import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    ToolUseBlock,
)

from backend.copilot import service as copilot_service
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.compaction import CompactionStats, CompactionTracker
from backend.copilot.sdk.service import _build_system_prompt_value
from backend.copilot.sdk.turn_summary import LANGFUSE_EVENT_NAME, TurnSummaryCollector
from backend.copilot.sdk.turn_summary_model import (
    TURN_SUMMARY_MARKER,
    TurnSummary,
    TurnUsage,
    short_hash,
    upload_gate_reason,
)
from backend.copilot.session_kind import SessionKind

_MOD = "backend.copilot.sdk.turn_summary"


def _session(messages: int = 1) -> ChatSession:
    rows = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        for i in range(messages)
    ]
    return ChatSession(
        session_id="sess-1",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=rows,
    )


def _assistant(
    msg_id: str | None,
    *,
    cache_read: int | None = 100,
    tools: tuple[str, ...] = (),
    parent: str | None = None,
) -> AssistantMessage:
    content = [ToolUseBlock(id=t, name="mcp__copilot__x", input={}) for t in tools]
    usage = None if cache_read is None else {"cache_read_input_tokens": cache_read}
    return AssistantMessage(
        content=content or [TextBlock(text="hi")],
        model="claude-sonnet-4-6",
        parent_tool_use_id=parent,
        usage=usage,
        message_id=msg_id,
    )


def _finish(collector: TurnSummaryCollector, **overrides) -> None:
    kwargs = dict(
        error=None,
        session=_session(),
        options=None,
        compaction=None,
        transport="openrouter",
        model_requested="anthropic/claude-sonnet-4-6",
        model_answered="anthropic/claude-sonnet-4-6",
        e2b=False,
        usage=TurnUsage(input_tokens=10, cache_read_tokens=20, output_tokens=5),
        codex=False,
        cost_reconcile=False,
        upload_gate=None,
        ended_with_stream_error=False,
    )
    kwargs.update(overrides)
    collector.finish(**kwargs)


def _lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if TURN_SUMMARY_MARKER in r.getMessage()
    ]


class TestLine:
    def test_is_the_marker_then_one_line_of_compact_json(self):
        line = TurnSummary(session_id="s", turn=2, error_class="a\nb").to_line()

        marker, payload = line.split(" ", 1)
        assert marker == TURN_SUMMARY_MARKER
        assert "\n" not in line
        assert json.loads(payload) == {
            "session_id": "s",
            "turn": 2,
            "error_class": "a\nb",
        }
        assert ": " not in payload and ", " not in payload

    def test_unset_fields_are_omitted_not_null(self):
        assert TurnSummary().to_line() == f"{TURN_SUMMARY_MARKER} {{}}"

    def test_field_order_is_stable_whatever_order_fields_are_set(self):
        a = TurnSummary()
        a.outcome, a.session_id = "ok", "s"
        b = TurnSummary()
        b.session_id, b.outcome = "s", "ok"

        assert (
            a.to_line()
            == b.to_line()
            == '[turn-summary] {"session_id":"s","outcome":"ok"}'
        )


class TestCollector:
    def test_a_voice_turn_is_marked_from_the_turn_not_the_text(self):
        voice = TurnSummaryCollector("s", voice=True)
        plain = TurnSummaryCollector("s")

        assert voice.summary.voice is True
        assert plain.summary.voice is False

    def test_record_turn_reads_the_session_kind(self):
        c = TurnSummaryCollector("sess-1", is_user_message=True)
        session = _session(3)
        session.metadata.origin = "automation"
        session.metadata.builder_graph_id = "g1"

        c.record_turn(
            turn=2,
            message_id="m",
            session=session,
            permissions=None,
            envelope=None,
            kind=SessionKind.for_turn(session),
        )

        s = c.summary
        assert (s.turn, s.message_id, s.first_turn) == (2, "m", False)
        assert (s.unattended, s.builder_session, s.permission_filtered) == (
            True,
            True,
            False,
        )
        assert (s.sub_session, s.expert_session, s.dry_run) == (False, False, False)

    def test_record_query_reads_blocks_path_and_history(self):
        c = TurnSummaryCollector("s")
        c.summary.first_turn = False
        history = "<conversation_history>\nUser: a\n</conversation_history>"
        query = f"{history}\n\nNow, the user says:\nhi"

        c.note_history_rows(4, compressed=True)
        c.record_query(query, use_resume=True, covered_rows=6, session_rows=11)

        s = c.summary
        assert s.context_blocks == ["conversation_history"]
        assert s.context_largest_block == "conversation_history"
        assert s.context_largest_block_chars == len(history)
        assert (s.query_path, s.history_rows, s.history_compressed) == (
            "resume_gap",
            4,
            True,
        )
        assert (s.transcript_covered_rows, s.session_rows) == (6, 11)
        assert s.query_chars == len(query)

    def test_rebuilt_query_is_marked_and_counted(self):
        c = TurnSummaryCollector("s")
        c.summary.first_turn = False
        c.record_query("hi", use_resume=True, covered_rows=3, session_rows=4)
        c.record_query(
            "hi", use_resume=False, covered_rows=0, session_rows=4, rebuilt=True
        )

        assert c.summary.query_path == "rebuilt_no_resume_bare"
        assert c.summary.context_rebuilds == 1
        assert c.summary.history_rows == 0
        assert c.summary.context_largest_block_chars == 0

    def test_first_turn_without_history_is_fresh(self):
        c = TurnSummaryCollector("s")
        c.summary.first_turn = True
        c.record_query(
            "<user_context>\np\n</user_context>\n\nhi",
            use_resume=False,
            covered_rows=0,
            session_rows=1,
        )

        assert c.summary.query_path == "fresh"
        assert c.summary.context_blocks == ["user_context"]

    def test_model_calls_count_once_per_message_id(self):
        c = TurnSummaryCollector("s")
        c.note_cli_launch()
        c.observe_assistant(_assistant("a", tools=("t1",)))
        c.observe_assistant(_assistant("a", tools=("t2",)))
        c.observe_assistant(_assistant("b", cache_read=0))
        c.observe_assistant(_assistant("c", cache_read=None))
        c.observe_assistant(_assistant("d", parent="t1", cache_read=0))

        s = c.summary
        assert (s.model_calls, s.subagent_model_calls, s.tool_calls) == (3, 1, 2)
        assert s.calls_without_cache_read == 1

    def test_first_call_of_each_launch_is_not_a_cache_miss(self):
        c = TurnSummaryCollector("s")
        c.note_cli_launch()
        c.observe_assistant(_assistant("a", cache_read=0))
        c.note_cli_launch()
        c.observe_assistant(_assistant("b", cache_read=0))

        assert c.summary.calls_without_cache_read is None
        assert (c.summary.cli_launches, c.summary.retries) == (2, 1)

    def test_finish_reads_options_compaction_and_cost(self):
        c = TurnSummaryCollector("s")
        tracker = CompactionTracker()
        tracker.on_compact()
        options = ClaudeAgentOptions(
            system_prompt="abc",
            allowed_tools=["mcp__copilot__b", "Read", "mcp__copilot__a"],
        )

        _finish(c, options=options, compaction=tracker, usage=TurnUsage(cost_usd=0.5))

        s = c.summary
        assert (s.system_prompt_mode, s.system_prompt_chars) == ("replace", 3)
        assert s.system_prompt_sha == short_hash("abc")
        assert (s.tools_allowed, s.tools_builtin) == (3, 1)
        assert s.tools_sha == short_hash("Read\nmcp__copilot__a\nmcp__copilot__b")
        assert s.cli_compactions == 1
        assert (s.cost_usd, s.cost_source) == (0.5, "rate_card")
        assert (s.outcome, s.error_class) == ("ok", None)

    def test_finish_records_the_declared_built_ins_beside_the_allowed_count(self):
        """``tools_builtin`` counts the built-ins the hook lets through; the
        declared list is what the CLI shows the model, which can be fewer."""
        allowed = ["mcp__copilot__a", "Read", "Write", "Edit", "Glob", "Grep"]
        options = ClaudeAgentOptions(
            allowed_tools=[*allowed, "TodoWrite"], tools=["TodoWrite", "Glob", "Grep"]
        )
        c = TurnSummaryCollector("s")

        _finish(c, options=options)

        s = c.summary
        assert s.tools_builtin == 6
        assert s.builtin_tools_declared == ["Glob", "Grep", "TodoWrite"]

    def test_finish_reads_the_pinned_window_off_the_launch_env(self):
        """The window, trigger and flags are what the launch's env carried,
        beside the source the turn noted for it."""
        env = {
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "272000",
            "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "90",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "272000",
        }
        c = TurnSummaryCollector("s")
        c.note_window_source("account")

        _finish(c, options=ClaudeAgentOptions(env=env), transport="codex")

        s = c.summary
        assert (s.context_window, s.autocompact_pct) == (272_000, 90)
        assert (s.max_context_tokens, s.disable_1m_context) == (272_000, False)
        assert s.window_source == "account"

    def test_finish_omits_the_window_when_the_launch_pinned_none(self):
        c = TurnSummaryCollector("s")

        _finish(c, options=ClaudeAgentOptions(env={"OTHER": "1"}))

        line = c.summary.model_dump(exclude_none=True)
        assert not {"context_window", "autocompact_pct", "disable_1m_context"} & set(
            line
        )

    @pytest.mark.asyncio
    async def test_compaction_counts_agree_with_the_compaction_events(self):
        """Each closed CLI cycle and each pre-query compaction sends one
        ``copilot-compaction`` event; the summary's counts are those events,
        and a failed cycle is one whose event names no readable summary."""
        session = _session()
        tracker = CompactionTracker()
        events: list[dict] = []
        with patch(
            "backend.copilot.sdk.compaction.emit_compaction_event",
            side_effect=lambda **kwargs: events.append(kwargs),
        ):
            for after_source in ("read", "no_summary_line", "read"):
                tracker.on_compact("/tmp/projects/abc/session.jsonl")
                tracker.emit_start_if_ready()
                await tracker.emit_end_if_ready(
                    session, CompactionStats(), after_source=after_source
                )
            tracker.on_compact("/tmp/projects/abc/session.jsonl")  # never closed
            tracker.emit_pre_query_end(session, CompactionStats())
        c = TurnSummaryCollector("s")

        _finish(c, session=session, compaction=tracker)

        s = c.summary
        cli = [e for e in events if e["path"] == "sdk_internal"]
        failed = [e["after_source"] for e in cli if e["after_source"] != "read"]
        assert (s.cli_compactions_landed, s.cli_compactions_failed) == (2, 1)
        assert s.cli_compactions_landed + s.cli_compactions_failed == len(cli)
        assert s.cli_compaction_failures == failed == ["no_summary_line"]
        assert s.pre_query_compactions == sum(e["path"] == "pre_query" for e in events)
        assert s.cli_compactions == 4

    def test_codex_usage_and_resend_rounds_ride_the_summary(self):
        c = TurnSummaryCollector("s")
        c.note_codex_usage(
            input_tokens=900, cached_input_tokens=600, boundary_peak_tokens=172_000
        )
        c.note_after_compaction_resend()
        c.note_after_compaction_resend()

        _finish(c, transport="codex", codex=True)

        s = c.summary
        assert (s.codex_input_tokens, s.codex_cached_input_tokens) == (900, 600)
        assert s.codex_boundary_peak_tokens == 172_000
        assert s.after_compaction_resends == 2

    def test_finish_omits_the_declared_built_ins_when_the_launch_set_none(self):
        c = TurnSummaryCollector("s")

        _finish(c, options=ClaudeAgentOptions(allowed_tools=["Read"]))

        assert "builtin_tools_declared" not in c.summary.model_dump(exclude_none=True)

    @pytest.mark.parametrize("live", [True, False])
    def test_finish_reads_the_prompt_in_every_form_a_launch_sends(
        self, monkeypatch, live
    ):
        monkeypatch.setattr(
            copilot_service, "_cached_prompt_identity", ("v38 prompt", 38)
        )
        prompt = "v38 prompt\nsupplement"
        options = ClaudeAgentOptions(
            system_prompt=_build_system_prompt_value(prompt, live=live)
        )
        c = TurnSummaryCollector("s")

        _finish(c, options=options)

        s = c.summary
        assert (s.system_prompt_mode, s.system_prompt_chars) == ("replace", len(prompt))
        assert s.system_prompt_sha == short_hash(prompt)
        assert (s.prompt_name, s.prompt_version) == (
            copilot_service.config.langfuse_prompt_name,
            38,
        )

    @pytest.mark.parametrize(
        "overrides, expected",
        [
            (dict(codex=True), "codex_subscription"),
            (dict(cost_reconcile=True), "openrouter_reconcile_pending"),
            (
                dict(
                    model_answered="moonshotai/kimi-k2.6", usage=TurnUsage(cost_usd=1.0)
                ),
                "moonshot_rate_card_estimate",
            ),
            (dict(usage=TurnUsage(cost_usd=1.0)), "rate_card"),
            (dict(), "none"),
        ],
    )
    def test_cost_source(self, overrides, expected):
        c = TurnSummaryCollector("s")
        _finish(c, **overrides)
        assert c.summary.cost_source == expected

    @pytest.mark.parametrize(
        "error, code, stream_error, expected",
        [
            (asyncio.CancelledError(), None, False, ("cancelled", "CancelledError")),
            (GeneratorExit(), None, False, ("cancelled", "GeneratorExit")),
            (ValueError("x"), None, False, ("error", "ValueError")),
            (None, "idle_timeout", True, ("error", "idle_timeout")),
            (None, None, True, ("error", "stream_error")),
        ],
    )
    def test_outcome(self, error, code, stream_error, expected):
        c = TurnSummaryCollector("s")
        if code:
            c.note_error(code)
        _finish(c, error=error, ended_with_stream_error=stream_error)
        assert (c.summary.outcome, c.summary.error_class) == expected

    def test_upload_fields(self):
        sent = TurnSummaryCollector("s")
        sent.note_upload(b"12345")
        missing = TurnSummaryCollector("s")
        missing.note_upload(None)
        gated = TurnSummaryCollector("s")
        _finish(gated, upload_gate="stream_error")

        assert sent.summary.transcript_upload_bytes == 5
        assert missing.summary.transcript_upload_skipped == "no_session_file"
        assert gated.summary.transcript_upload_skipped == "stream_error"


class TestNeverRaises:
    def test_a_bad_input_is_logged_and_dropped(self, caplog):
        c = TurnSummaryCollector("s")
        with caplog.at_level(logging.WARNING, logger=_MOD):
            c.record_turn(
                turn=1,
                message_id="m",
                session=MagicMock(metadata=None),
                permissions=None,
                envelope=None,
                kind=MagicMock(),
            )
            c.observe_assistant(MagicMock(content=None))
            _finish(c, session=MagicMock(spec_set=[]))

        assert {"record_turn", "observe_assistant", "finish"} <= {
            r.args[0] for r in caplog.records if r.args
        }

    def test_emit_survives_a_failing_serialiser_and_langfuse(self, caplog):
        c = TurnSummaryCollector("s")
        with (
            patch.object(TurnSummary, "to_line", side_effect=RuntimeError("boom")),
            caplog.at_level(logging.INFO, logger=_MOD),
        ):
            c.emit("trace-1")
        assert _lines(caplog) == []

        d = TurnSummaryCollector("s")
        with (
            patch(f"{_MOD}.get_client", side_effect=RuntimeError("down")),
            caplog.at_level(logging.INFO, logger=_MOD),
        ):
            d.emit("trace-1")
        assert len(_lines(caplog)) == 1


class TestEmit:
    def test_logs_once_and_attaches_the_same_dict_to_the_trace(self, caplog):
        c = TurnSummaryCollector("s", is_user_message=True)
        _finish(c)
        client = MagicMock()
        with (
            patch(f"{_MOD}.get_client", return_value=client),
            caplog.at_level(logging.INFO, logger=_MOD),
        ):
            c.emit("trace-1")
            c.emit("trace-1")

        lines = _lines(caplog)
        assert len(lines) == 1
        client.create_event.assert_called_once()
        kwargs = client.create_event.call_args.kwargs
        assert kwargs["name"] == LANGFUSE_EVENT_NAME
        assert kwargs["trace_context"] == {"trace_id": "trace-1"}
        assert kwargs["metadata"] == json.loads(lines[0].split(" ", 1)[1])

    def test_no_trace_id_logs_without_an_event(self, caplog):
        c = TurnSummaryCollector("s")
        with (
            patch(f"{_MOD}.get_client") as get_client,
            caplog.at_level(logging.INFO, logger=_MOD),
        ):
            c.emit(None)
        get_client.assert_not_called()
        assert len(_lines(caplog)) == 1


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(user_id="u", started=True, ended_with_stream_error=False), None),
        (dict(user_id=None, started=True, ended_with_stream_error=False), "anonymous"),
        # ``started`` is False whenever the cwd, the session or the retry state
        # was never set up: the upload needs all three.
        (
            dict(user_id="u", started=False, ended_with_stream_error=False),
            "not_started",
        ),
        (dict(user_id="u", started=True, ended_with_stream_error=True), "stream_error"),
    ],
)
def test_upload_gate_reason_names_why_the_upload_is_skipped(kwargs, expected):
    """The turn evaluates this once and both the CLI session upload and the
    summary's ``transcript_upload_skipped`` read the answer."""
    assert upload_gate_reason(**kwargs) == expected
