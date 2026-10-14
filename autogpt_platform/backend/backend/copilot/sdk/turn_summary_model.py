"""The turn-summary line's fields, and the pure functions that derive them.

See :mod:`backend.copilot.sdk.turn_summary` for the collector that fills a
:class:`TurnSummary` in as a turn runs.
"""

import asyncio
import hashlib
import json

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk.types import (
    SystemPromptCustom,
    SystemPromptFile,
    SystemPromptPreset,
)
from pydantic import BaseModel

from backend.copilot.moonshot import is_moonshot_model
from backend.copilot.service import system_prompt_source

TURN_SUMMARY_MARKER = "[turn-summary]"
_HASH_CHARS = 12


class TurnSummary(BaseModel):
    """The fields of one turn-summary line. Unset fields are omitted."""

    # identity
    session_id: str | None = None
    message_id: str | None = None
    turn: int | None = None
    user_message: bool | None = None
    first_turn: bool | None = None
    origin: str | None = None
    session_kind: str | None = None
    builder_session: bool | None = None
    expert_session: bool | None = None
    expert_kickoff: bool | None = None
    sub_session: bool | None = None
    delegated: bool | None = None
    unattended: bool | None = None
    permission_filtered: bool | None = None
    voice: bool | None = None
    e2b: bool | None = None
    dry_run: bool | None = None
    # route
    transport: str | None = None
    model_requested: str | None = None
    model_answered: str | None = None
    release: str | None = None
    # context window the CLI was pinned to, read off the launch's env, and
    # where the window came from (``sdk/context_window.py``)
    context_window: int | None = None
    autocompact_pct: int | None = None
    window_source: str | None = None
    max_context_tokens: int | None = None
    disable_1m_context: bool | None = None
    # prompt
    prompt_name: str | None = None
    prompt_version: int | None = None
    system_prompt_mode: str | None = None
    system_prompt_sha: str | None = None
    system_prompt_chars: int | None = None
    building_guide_in_prompt: bool | None = None
    # tools: counts and hash over allowed_tools (what the hook lets through),
    # then the built-ins the CLI declares to the model (options.tools)
    tools_allowed: int | None = None
    tools_builtin: int | None = None
    tools_sha: str | None = None
    builtin_tools_declared: list[str] | None = None
    # context blocks on the query the last CLI launch was sent
    context_blocks: list[str] | None = None
    context_prefix_chars: int | None = None
    context_largest_block: str | None = None
    context_largest_block_chars: int | None = None
    query_chars: int | None = None
    # conversation
    query_path: str | None = None
    context_rebuilds: int | None = None
    transcript_covered_rows: int | None = None
    history_rows: int | None = None
    history_compressed: bool | None = None
    session_rows: int | None = None
    transcript_download_bytes: int | None = None
    transcript_upload_bytes: int | None = None
    transcript_upload_skipped: str | None = None
    # turn shape
    cli_launches: int | None = None
    retries: int | None = None
    model_calls: int | None = None
    subagent_model_calls: int | None = None
    tool_calls: int | None = None
    # compactions: the CLI's (started, then how the closed ones ended, as the
    # ``copilot-compaction`` events report them), the copilot's own, and the
    # first-turn context re-sent after a CLI compaction
    cli_compactions: int | None = None
    cli_compactions_landed: int | None = None
    cli_compactions_failed: int | None = None
    cli_compaction_failures: list[str] | None = None
    pre_query_compactions: int | None = None
    after_compaction_resends: int | None = None
    # tokens and cost
    input_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    calls_without_cache_read: int | None = None
    cost_usd: float | None = None
    cost_source: str | None = None
    # the Codex gateway's own counts: a turn total over every invoke, and the
    # largest request it sized
    codex_input_tokens: int | None = None
    codex_cached_input_tokens: int | None = None
    codex_boundary_peak_tokens: int | None = None
    # outcome
    outcome: str | None = None
    error_class: str | None = None

    def to_line(self) -> str:
        """The log line: the marker, then one line of compact JSON."""
        payload = json.dumps(
            self.model_dump(exclude_none=True),
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return f"{TURN_SUMMARY_MARKER} {payload}"


class TurnUsage(BaseModel):
    """Token counts and cost as the turn already recorded them."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


def upload_gate_reason(
    *,
    user_id: str | None,
    started: bool,
    ended_with_stream_error: bool,
) -> str | None:
    """Why the end of the turn skips the CLI session upload, or ``None``.

    ``stream_chat_completion_sdk`` evaluates it once: the upload runs only
    when it is ``None``, and the turn summary reports what it returned.
    """
    if not user_id:
        return "anonymous"
    if not started:
        return "not_started"
    if ended_with_stream_error:
        return "stream_error"
    return None


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_CHARS]


def query_path(*, use_resume: bool, first_turn: bool, history_sent: bool) -> str:
    """How the query reached the CLI: resumed, fresh, or rebuilt from history."""
    if use_resume:
        return "resume_gap" if history_sent else "resume"
    if first_turn:
        return "fresh"
    return "no_resume_history" if history_sent else "no_resume_bare"


def record_usage(
    s: TurnSummary,
    usage: TurnUsage,
    *,
    model: str | None,
    codex: bool,
    cost_reconcile: bool,
) -> None:
    s.input_tokens = usage.input_tokens
    s.cache_read_tokens = usage.cache_read_tokens
    s.cache_write_tokens = usage.cache_write_tokens
    s.output_tokens = usage.output_tokens
    s.cost_usd = usage.cost_usd
    s.cost_source = cost_source(
        usage.cost_usd, model=model, codex=codex, cost_reconcile=cost_reconcile
    )


def cost_source(
    cost_usd: float | None, *, model: str | None, codex: bool, cost_reconcile: bool
) -> str:
    """Where ``cost_usd`` came from.

    ``rate_card``: the platform's rate card priced the tokens the provider
    reported; the CLI's own figure is never the cost.
    ``openrouter_reconcile_pending``: the provider's bill replaces the rate
    card's figure later, in the ``openrouter-cost-reconcile`` event and log
    line.
    """
    if codex:
        return "codex_subscription"
    if cost_reconcile:
        return "openrouter_reconcile_pending"
    if cost_usd is None:
        return "none"
    if is_moonshot_model(model):
        return "moonshot_rate_card_estimate"
    return "rate_card"


def record_options(s: TurnSummary, options: ClaudeAgentOptions) -> None:
    """System prompt identity and tool surface of the options the CLI ran with."""
    text, s.system_prompt_mode = _system_prompt_text(options.system_prompt)
    s.system_prompt_sha = short_hash(text)
    s.system_prompt_chars = len(text)
    s.prompt_name, s.prompt_version = system_prompt_source(text)
    tools = sorted(options.allowed_tools)
    s.tools_allowed = len(tools)
    s.tools_builtin = sum(1 for name in tools if not name.startswith("mcp__"))
    s.tools_sha = short_hash("\n".join(tools))
    if isinstance(options.tools, list):
        s.builtin_tools_declared = sorted(options.tools)
    env = options.env
    s.context_window = _env_int(env, "CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    if s.context_window is not None:
        s.autocompact_pct = _env_int(env, "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE")
        s.max_context_tokens = _env_int(env, "CLAUDE_CODE_MAX_CONTEXT_TOKENS")
        s.disable_1m_context = env.get("CLAUDE_CODE_DISABLE_1M_CONTEXT") == "1"


def _env_int(env: dict[str, str], name: str) -> int | None:
    """An integer the launch's env sets, or ``None`` when it sets none."""
    value = env.get(name, "")
    return int(value) if value.isdigit() else None


def _system_prompt_text(
    prompt: str | SystemPromptPreset | SystemPromptCustom | SystemPromptFile | None,
) -> tuple[str, str | None]:
    """The prompt text a launch was given, and how the CLI combines it with its own.

    Every launch replaces the CLI's prompt: with a raw string, or with the
    live prompt's ``custom`` dict, which can turn the CLI's snapshot off.
    """
    if isinstance(prompt, str):
        return prompt, "replace"
    if prompt is not None and prompt["type"] == "custom":
        return prompt["prompt"], "replace"
    return "", None


def outcome(
    error: BaseException | None, code: str | None, ended_with_stream_error: bool
) -> tuple[str, str | None]:
    """``(outcome, error_class)`` for the turn's end state."""
    if isinstance(error, (asyncio.CancelledError, GeneratorExit)):
        return "cancelled", type(error).__name__
    if error is not None:
        return "error", type(error).__name__
    if code is not None:
        return "error", code
    if ended_with_stream_error:
        return "error", "stream_error"
    return "ok", None
