"""The turn-summary line is emitted exactly once per SDK turn, whatever its end.

Drives the real ``stream_chat_completion_sdk`` with a mocked CLI client, on a
resumed session so the first-turn context injection (which reaches the DB)
does not run, and with every Redis or database call of that path stubbed.
"""

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
)

from backend.copilot import service as copilot_service
from backend.copilot.anthropic_rate_card import anthropic_usage_cost_usd
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.launch_env import isolated_launch_env
from backend.copilot.sdk.query_blocks import scan_context_blocks
from backend.copilot.sdk.retry_scenarios_test import _make_sdk_patches
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.sdk.turn_summary_model import TURN_SUMMARY_MARKER, short_hash

_SVC = "backend.copilot.sdk.service"
_SUMMARY_LOGGER = "backend.copilot.sdk.turn_summary"


def _session() -> ChatSession:
    return ChatSession(
        session_id="test-session-id",
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[
            ChatMessage(role="user", content="earlier question", sequence=0),
            ChatMessage(role="assistant", content="earlier answer", sequence=1),
            ChatMessage(role="user", content="hello", sequence=2),
        ],
    )


def _client(messages: list, query_error: BaseException | None = None) -> AsyncMock:
    async def _receive():
        for message in messages:
            yield message

    client = MagicMock()
    client.receive_response = _receive
    client.query = AsyncMock(side_effect=query_error)
    client._transport = MagicMock()
    client._transport.write = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm


def _answer() -> list:
    return [
        AssistantMessage(
            content=[TextBlock(text="hi there")],
            model="claude-sonnet-4-6",
            usage={"input_tokens": 12, "cache_read_input_tokens": 300},
            message_id="msg_1",
        ),
        ResultMessage(
            subtype="success",
            result="hi there",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=1,
            session_id="test-session-id",
            total_cost_usd=0.0123,
            usage={
                "input_tokens": 12,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 40,
                "output_tokens": 7,
            },
        ),
    ]


async def _run_turn(
    tmp_path: Path,
    client_factory,
    *,
    session: ChatSession | None = None,
    extra_patches: list | None = None,
    config_overrides: dict | None = None,
) -> list:
    session = session or _session()
    transcript = build_test_transcript(
        [("user", "earlier question"), ("assistant", "earlier answer")]
    )
    patches = _make_sdk_patches(session, transcript, None, client_factory) + [
        (f"{_SVC}._make_sdk_cwd", dict(return_value=str(tmp_path))),
        (f"{_SVC}.resolve_answered", dict(new_callable=AsyncMock, return_value=[])),
        (f"{_SVC}._update_title_async", dict(new_callable=AsyncMock)),
        (
            f"{_SVC}.build_turn_budget_block",
            dict(new_callable=AsyncMock, return_value=""),
        ),
        (
            f"{_SVC}.build_skills_update_notice",
            dict(new_callable=AsyncMock, return_value=""),
        ),
        (f"{_SVC}.persist_and_record_usage", dict(new_callable=AsyncMock)),
    ]
    events = []
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches + (extra_patches or []):
            stack.enter_context(patch(target, **kwargs))
        for name, value in (config_overrides or {}).items():
            setattr(sdk_service.config, name, value)
        with contextlib.suppress(Exception, asyncio.CancelledError):
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)
    return events


def _launches(*scripts) -> tuple:
    """A client factory that plays one script per CLI launch, the options each
    launch was given, and the client each launch got."""
    options: list[ClaudeAgentOptions] = []
    clients: list[AsyncMock] = []

    def factory(*args, **kwargs):
        options.append(kwargs["options"])
        clients.append(scripts[len(options) - 1]())
        return clients[-1]

    return factory, options, clients


def _sent_query(client: AsyncMock) -> str:
    """The query a launch sent to its CLI."""
    return client.__aenter__.return_value.query.await_args.args[0]


def _summaries(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        json.loads(r.getMessage().split(" ", 1)[1])
        for r in caplog.records
        if r.name == _SUMMARY_LOGGER and r.getMessage().startswith(TURN_SUMMARY_MARKER)
    ]


def _assert_line_reads_the_query(summary: dict, query: str, path: str) -> None:
    """The line's query fields describe *query*, the last launch's."""
    blocks, prefix_chars = scan_context_blocks(query)
    largest = max(blocks, key=lambda block: block[1], default=None)
    assert summary["query_chars"] == len(query)
    assert summary["query_path"] == path
    assert summary["context_blocks"] == [name for name, _ in blocks]
    assert summary["context_prefix_chars"] == prefix_chars
    assert summary.get("context_largest_block") == (largest and largest[0])


@pytest.mark.asyncio
async def test_a_successful_turn_logs_one_summary(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(tmp_path, lambda *a, **k: _client(_answer()))

    [summary] = _summaries(caplog)
    assert summary["outcome"] == "ok"
    assert summary["session_id"] == "test-session-id"
    assert (summary["turn"], summary["first_turn"]) == (2, False)
    assert (summary["cli_launches"], summary["retries"]) == (1, 0)
    assert (summary["model_calls"], summary["tool_calls"]) == (1, 0)
    assert summary["model_answered"] == "claude-sonnet-4-6"
    assert (summary["input_tokens"], summary["cache_read_tokens"]) == (12, 300)
    assert (summary["cache_write_tokens"], summary["output_tokens"]) == (40, 7)
    # Priced on the rate card from the answer's tokens, not the CLI's $0.0123.
    assert summary["cost_source"] == "rate_card"
    assert summary["cost_usd"] == pytest.approx(
        anthropic_usage_cost_usd(
            model="claude-sonnet-4-6",
            input_tokens=12,
            output_tokens=7,
            cache_read_tokens=300,
            cache_write_5m_tokens=40,
        )
    )
    assert summary["transcript_download_bytes"] > 0
    assert summary["transcript_covered_rows"] == 2
    assert summary["session_rows"] == 3
    assert summary["query_path"] == "no_resume_history"
    assert summary["transcript_upload_skipped"] == "no_session_file"
    assert summary["system_prompt_chars"] > 0
    assert "error_class" not in summary


@pytest.mark.asyncio
async def test_a_failing_turn_logs_one_summary(tmp_path, caplog):
    def _failing(*args, **kwargs):
        return _client([], query_error=RuntimeError("authentication failed"))

    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(tmp_path, _failing)

    [summary] = _summaries(caplog)
    assert summary["outcome"] == "error"
    assert summary["error_class"]
    assert summary["cli_launches"] == 1
    assert summary["transcript_upload_skipped"] == "stream_error"


@pytest.mark.asyncio
async def test_a_cancelled_turn_logs_one_summary(tmp_path, caplog):
    def _cancelled(*args, **kwargs):
        return _client([], query_error=asyncio.CancelledError())

    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(tmp_path, _cancelled)

    [summary] = _summaries(caplog)
    assert (summary["outcome"], summary["error_class"]) == (
        "cancelled",
        "CancelledError",
    )


@pytest.mark.asyncio
async def test_a_turn_whose_cleanup_raises_still_logs_one_summary(tmp_path, caplog):
    """The usage persist after the turn raises; the line still comes out, with
    the turn's own outcome and tokens."""
    persist = AsyncMock(side_effect=RuntimeError("usage store down"))

    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(
            tmp_path,
            lambda *a, **k: _client(_answer()),
            extra_patches=[(f"{_SVC}.persist_and_record_usage", dict(new=persist))],
        )

    persist.assert_awaited_once()
    [summary] = _summaries(caplog)
    assert (summary["outcome"], summary["cli_launches"]) == ("ok", 1)
    assert (summary["input_tokens"], summary["output_tokens"]) == (12, 7)


# Every CLI launch of a turn (the first, a context-size retry, a building-mode
# restart) must keep the first launch's hooks (the after-compaction re-send)
# and route environment, send the live prompt, and leave a summary line that
# hashes and versions the prompt the last launch was given and lists the
# built-ins it declared.

_GUIDE = "\n\n<building_guide>GUIDE</building_guide>"
_ROUTE_ENV = {
    "ANTHROPIC_BASE_URL": "https://openrouter.test/api",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "65",
    "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1",
}
_ALLOWED = ["mcp__copilot__find_capability", "Glob", "TodoWrite"]
_DISALLOWED = ["Bash", "Glob"]
_LIVE_PROMPT = {"claude_agent_live_system_prompt": True}


@pytest.fixture
def langfuse_prompt_v38(monkeypatch):
    monkeypatch.setattr(
        copilot_service, "_cached_prompt_identity", ("system prompt", 38)
    )


def _launch_patches() -> list:
    return [
        (f"{_SVC}.route_launch_env", dict(return_value=dict(_ROUTE_ENV))),
        (f"{_SVC}.get_copilot_tool_names", dict(return_value=list(_ALLOWED))),
        (f"{_SVC}.get_sdk_disallowed_tools", dict(return_value=list(_DISALLOWED))),
    ]


def _prompt_text(options: ClaudeAgentOptions) -> str:
    prompt = options.system_prompt
    assert isinstance(prompt, dict) and prompt["type"] == "custom"
    assert prompt.get("snapshot") is False
    return prompt["prompt"]


def _assert_line_reads_the_launch(summary: dict, options: ClaudeAgentOptions) -> None:
    text = _prompt_text(options)
    assert summary["system_prompt_mode"] == "replace"
    assert summary["system_prompt_sha"] == short_hash(text)
    assert summary["system_prompt_chars"] == len(text)
    assert summary["prompt_version"] == 38
    assert options.tools == ["TodoWrite"]
    assert (options.allowed_tools, options.disallowed_tools) == (
        _ALLOWED,
        _DISALLOWED,
    )
    assert summary["builtin_tools_declared"] == ["TodoWrite"]
    # The pin the launch's env carried, read back off it.
    assert (summary["context_window"], summary["autocompact_pct"]) == (200_000, 65)
    assert summary["disable_1m_context"] is True


@pytest.mark.asyncio
async def test_first_launch_line_reads_the_live_prompt(
    tmp_path, caplog, langfuse_prompt_v38
):
    factory, options, clients = _launches(lambda: _client(_answer()))

    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(
            tmp_path,
            factory,
            extra_patches=_launch_patches(),
            config_overrides=_LIVE_PROMPT,
        )

    [summary] = _summaries(caplog)
    [launch] = options
    # The route's own variables, launched from the allowlisted parent.
    assert launch.env == isolated_launch_env(_ROUTE_ENV)
    _assert_line_reads_the_launch(summary, launch)
    _assert_line_reads_the_query(summary, _sent_query(clients[0]), "no_resume_history")


@pytest.mark.asyncio
async def test_context_size_retry_keeps_hooks_env_and_live_prompt(
    tmp_path, caplog, langfuse_prompt_v38
):
    factory, options, clients = _launches(
        lambda: _client(
            [], query_error=Exception("prompt is too long (context_length_exceeded)")
        ),
        lambda: _client(_answer()),
    )

    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(
            tmp_path,
            factory,
            extra_patches=_launch_patches()
            + [
                (
                    f"{_SVC}._rebuild_first_turn_context",
                    dict(new_callable=AsyncMock, return_value=[]),
                )
            ],
            config_overrides=_LIVE_PROMPT,
        )

    [summary] = _summaries(caplog)
    first, retry = options
    assert (summary["cli_launches"], summary["retries"]) == (2, 1)
    assert summary["context_rebuilds"] == 1
    _assert_line_reads_the_query(
        summary, _sent_query(clients[1]), "rebuilt_no_resume_history"
    )
    assert retry.hooks is first.hooks
    assert retry.env == first.env == isolated_launch_env(_ROUTE_ENV)
    assert _prompt_text(retry) == _prompt_text(first)
    _assert_line_reads_the_launch(summary, retry)


@pytest.mark.asyncio
async def test_building_mode_restart_keeps_hooks_env_and_sends_the_guide(
    tmp_path, caplog, langfuse_prompt_v38
):
    session = _session()
    session.building_mode_requested = True
    factory, options, clients = _launches(
        lambda: _client(_answer()[:1]), lambda: _client(_answer())
    )

    async def _builder_suffix(_session, force=False):
        return _GUIDE if force else ""

    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(
            tmp_path,
            factory,
            session=session,
            extra_patches=_launch_patches()
            + [
                (
                    f"{_SVC}.build_builder_system_prompt_suffix",
                    dict(side_effect=_builder_suffix),
                )
            ],
            config_overrides=_LIVE_PROMPT,
        )

    [summary] = _summaries(caplog)
    first, restart = options
    assert summary["cli_launches"] == 2
    assert summary["building_guide_in_prompt"] is True
    assert restart.hooks is first.hooks
    assert restart.env == first.env == isolated_launch_env(_ROUTE_ENV)
    assert _prompt_text(restart) == _prompt_text(first) + _GUIDE
    _assert_line_reads_the_launch(summary, restart)
    _assert_line_reads_the_query(summary, _sent_query(clients[1]), "resume")
