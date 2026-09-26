from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tiktoken import encoding_for_model, encoding_name_for_model

from backend.data.llm_registry.llm_models import (
    CLAUDE_5_TOKENIZER_GENERATION_PREFIXES,
    strip_anthropic_vendor_prefix,
)
from backend.util import json

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
#  CONSTANTS                                                                 #
# ---------------------------------------------------------------------------#

# Message prefixes for important system messages that should be protected during compression
MAIN_OBJECTIVE_PREFIX = "[Main Objective Prompt]: "

# ---------------------------------------------------------------------------#
#  INTERNAL UTILITIES                                                         #
# ---------------------------------------------------------------------------#


def _tok_len(text: str, enc) -> int:
    """True token length of *text* in tokenizer *enc* (no wrapper cost)."""
    return len(enc.encode(str(text)))


def _msg_tokens(msg: dict, enc) -> int:
    """
    OpenAI counts ≈3 wrapper tokens per chat message, plus 1 if "name"
    is present, plus the tokenised content length.
    For tool calls, we need to count tokens in tool_calls and content fields.
    Supports Chat Completions, Anthropic, and Responses API formats.
    """
    WRAPPER = 3 + (1 if "name" in msg else 0)

    # Responses API: function_call items have arguments + name
    if msg.get("type") == "function_call":
        return (
            WRAPPER
            + _tok_len(msg.get("name", ""), enc)
            + _tok_len(msg.get("arguments", ""), enc)
            + _tok_len(msg.get("call_id", ""), enc)
        )

    # Responses API: function_call_output items have output
    if msg.get("type") == "function_call_output":
        return (
            WRAPPER
            + _tok_len(msg.get("output", ""), enc)
            + _tok_len(msg.get("call_id", ""), enc)
        )

    # Count content tokens.  List content is counted block by block below;
    # tokenizing str(list) here first would encode a base64 image twice.
    content = msg.get("content")
    content_tokens = 0 if isinstance(content, list) else _tok_len(content or "", enc)

    # Count tool call tokens for both OpenAI and Anthropic formats
    tool_call_tokens = 0

    # OpenAI Chat Completions format: tool_calls array at message level
    if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
        for tool_call in msg["tool_calls"]:
            # Count the tool call structure tokens
            tool_call_tokens += _tok_len(tool_call.get("id", ""), enc)
            tool_call_tokens += _tok_len(tool_call.get("type", ""), enc)
            if "function" in tool_call:
                tool_call_tokens += _tok_len(tool_call["function"].get("name", ""), enc)
                tool_call_tokens += _tok_len(
                    tool_call["function"].get("arguments", ""), enc
                )

    # Anthropic format: tool_use within content array
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                # Count the tool use structure tokens
                tool_call_tokens += _tok_len(item.get("id", ""), enc)
                tool_call_tokens += _tok_len(item.get("name", ""), enc)
                tool_call_tokens += _tok_len(json.dumps(item.get("input", {})), enc)
            elif isinstance(item, dict) and item.get("type") == "tool_result":
                # Count tool result tokens
                tool_call_tokens += _tok_len(item.get("tool_use_id", ""), enc)
                tool_call_tokens += _tok_len(item.get("content", ""), enc)
            elif isinstance(item, dict) and item.get("type") == "text":
                # Count text block tokens (standard: "text" key, fallback: "content")
                text_val = item.get("text") or item.get("content", "")
                tool_call_tokens += _tok_len(text_val, enc)
            elif isinstance(item, dict) and "content" in item:
                # Other content types with content field
                tool_call_tokens += _tok_len(item.get("content", ""), enc)
            elif isinstance(item, dict):
                # Images, documents and any other block: count what will be
                # sent.  Ignoring them let a 1 MB image register as 8 tokens
                # and pass through compaction whole.  Estimated from the
                # serialised length rather than tokenized: BPE over a megabyte
                # of base64 takes seconds, and ~3 characters per token is the
                # conservative side for that alphabet.
                tool_call_tokens += len(json.dumps(item, separators=(",", ":"))) // 3

    return WRAPPER + content_tokens + tool_call_tokens


def _is_tool_message(msg: dict) -> bool:
    """Check if a message contains tool calls or results that should be protected."""
    # Responses API: standalone function_call / function_call_output items
    if msg.get("type") in ("function_call", "function_call_output"):
        return True

    content = msg.get("content")

    # Check for Anthropic-style tool messages
    if isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type") in ("tool_use", "tool_result")
        for item in content
    ):
        return True

    # Check for OpenAI Chat Completions-style tool calls in the message
    if "tool_calls" in msg or msg.get("role") == "tool":
        return True

    return False


def _is_objective_message(msg: dict) -> bool:
    """Check if a message contains objective/system prompts that should be absolutely protected."""
    content = msg.get("content", "")
    if isinstance(content, str):
        # Protect any message with the main objective prefix
        return content.startswith(MAIN_OBJECTIVE_PREFIX)
    return False


def _truncate_tool_message_content(msg: dict, enc, max_tokens: int) -> None:
    """
    Carefully truncate tool message content while preserving tool structure.
    Handles Anthropic, Chat Completions, and Responses API tool messages.
    """
    # Responses API: function_call_output has "output" field
    if msg.get("type") == "function_call_output":
        output = msg.get("output", "")
        if isinstance(output, str) and _tok_len(output, enc) > max_tokens:
            msg["output"] = _truncate_middle_tokens(output, enc, max_tokens)
        return

    content = msg.get("content")

    # OpenAI Chat Completions tool message: role="tool" with string content
    if msg.get("role") == "tool" and isinstance(content, str):
        if _tok_len(content, enc) > max_tokens:
            msg["content"] = _truncate_middle_tokens(content, enc, max_tokens)
        return

    # Anthropic-style: list content with tool_result items
    if not isinstance(content, list):
        return

    for item in content:
        # Only process tool_result items, leave tool_use blocks completely intact
        if not (isinstance(item, dict) and item.get("type") == "tool_result"):
            continue

        result_content = item.get("content", "")
        if (
            isinstance(result_content, str)
            and _tok_len(result_content, enc) > max_tokens
        ):
            item["content"] = _truncate_middle_tokens(result_content, enc, max_tokens)


def _truncate_middle_tokens(text: str, enc, max_tok: int) -> str:
    """
    Return *text* shortened to ≈max_tok tokens by keeping the head & tail
    and inserting an ellipsis token in the middle.
    """
    ids = enc.encode(str(text))
    if len(ids) <= max_tok:
        return text  # nothing to do

    # Need at least 3 tokens (head + ellipsis + tail) for meaningful truncation
    if max_tok < 1:
        return ""
    mid = enc.encode(" … ")
    if max_tok < 3:
        return enc.decode(ids[:max_tok])

    # Split the allowance between the two ends:
    head = max_tok // 2 - 1  # -1 for the ellipsis
    tail = max_tok - head - 1
    return enc.decode(ids[:head] + mid + ids[-tail:])


# ---------------------------------------------------------------------------#
#  PUBLIC API                                                                 #
# ---------------------------------------------------------------------------#


def estimate_token_count(
    messages: list[dict],
    *,
    model: str = "gpt-4o",
) -> int:
    """
    Return the true token count of *messages* when encoded for *model*.

    Parameters
    ----------
    messages    Complete chat history.
    model       Model name; passed to tiktoken to pick the right
                tokenizer (gpt-4o → 'o200k_base', others fallback).

    Returns
    -------
    int  – Token count.
    """
    token_model = _normalize_model_for_tokenizer(model)
    enc = encoding_for_model(token_model)
    raw = sum(_msg_tokens(m, enc) for m in messages)
    return int(raw * _token_estimate_factor(model))


def estimate_token_count_str(
    text: Any,
    *,
    model: str = "gpt-4o",
) -> int:
    """
    Return the true token count of *text* when encoded for *model*.

    Parameters
    ----------
    text    Input text.
    model   Model name; passed to tiktoken to pick the right
            tokenizer (gpt-4o → 'o200k_base', others fallback).

    Returns
    -------
    int  – Token count.
    """
    token_model = _normalize_model_for_tokenizer(model)
    enc = encoding_for_model(token_model)
    text = json.dumps(text) if not isinstance(text, str) else text
    return int(_tok_len(text, enc) * _token_estimate_factor(model))


# ---------------------------------------------------------------------------#
#  UNIFIED CONTEXT COMPRESSION                                               #
# ---------------------------------------------------------------------------#

# Default thresholds
DEFAULT_TOKEN_THRESHOLD = 120_000
DEFAULT_KEEP_RECENT = 15

# Response headroom subtracted from the compression target.  Named (rather than
# inlined in ``compress_context``'s signature) because the copilot's pre-query
# predictor mirrors the early-return condition below and must read the same
# value — a silent desync there opens compaction rows for work that never runs.
DEFAULT_COMPRESSION_RESERVE = 2_048

# Reserve tokens for system prompt, tool definitions, and per-turn overhead.
# The actual model context limit minus this reserve = compression target.
_CONTEXT_OVERHEAD_RESERVE = 60_000


def get_context_window(model: str) -> int | None:
    """Return the context window size for a model, or None if unknown.

    Looks up the model in the :class:`LLMModel` enum (which already
    carries ``context_window`` via ``MODEL_METADATA``).  Handles
    provider-prefixed names (``anthropic/claude-opus-4-6``) and
    case-insensitive input automatically.
    """
    from backend.blocks.llm import LLMModel  # lazy to avoid circular import

    try:
        llm_model = LLMModel(model)
        return llm_model.context_window
    except (ValueError, KeyError):
        pass

    # Retry with lowercase for case-insensitive lookup
    try:
        llm_model = LLMModel(model.lower())
        return llm_model.context_window
    except (ValueError, KeyError):
        return None


def get_compression_target(model: str) -> int:
    """Compute a model-aware compression target for conversation history.

    Returns ``context_window - overhead_reserve``, floored at 10K.
    Falls back to ``DEFAULT_TOKEN_THRESHOLD`` for unknown models or
    models whose context window is too small for the overhead reserve.
    """
    window = get_context_window(model)
    if window is None:
        return DEFAULT_TOKEN_THRESHOLD
    target = window - _CONTEXT_OVERHEAD_RESERVE
    if target < 10_000:
        return DEFAULT_TOKEN_THRESHOLD
    return target


@dataclass
class CompressResult:
    """Result of context compression."""

    messages: list[dict]
    token_count: int
    was_compacted: bool
    error: str | None = None
    original_token_count: int = 0
    messages_summarized: int = 0
    messages_dropped: int = 0
    # Fraction of the summarised history the summariser actually read (1.0
    # unless it exceeded MAX_SUMMARY_CHUNKS).  Older code silently read the
    # first 100K characters and reported every message as summarised.
    summary_coverage: float = 1.0
    # False when no client was available: the result is truncation only,
    # with no summary of anything that was dropped.
    summarizer_available: bool = True
    # Tokens of the most recent history kept verbatim beside the summary.
    tail_tokens: int = 0


# Estimation correction for the Claude 5 family (sonnet-5/fable-5/
# mythos-5, plus the shared 4.7/4.8 tokenizer generation): Anthropic ships
# no local tokenizer, so estimates ride tiktoken o200k_base, which already
# undercounts Claude 4.x by ~15-20%; the Claude 5 tokenizer additionally
# counts ~30% MORE tokens for the same text (Anthropic migration guide).
# 1.5 ≈ 1.175 x 1.3 — deliberately on the high side: overestimating
# shrinks compaction targets and max_tokens headroom (safe), while
# underestimating overflows the context window (a 400 at serve time).
CLAUDE_5_TOKEN_FACTOR = 1.5


def _token_estimate_factor(model: str) -> float:
    if strip_anthropic_vendor_prefix(model).startswith(
        CLAUDE_5_TOKENIZER_GENERATION_PREFIXES
    ):
        return CLAUDE_5_TOKEN_FACTOR
    return 1.0


def _normalize_model_for_tokenizer(model: str) -> str:
    """Normalize model name for tiktoken tokenizer selection."""
    if "/" in model:
        model = model.split("/")[-1]
    if "claude" in model.lower() or not any(
        known in model.lower() for known in ["gpt", "o1", "chatgpt", "text-"]
    ):
        return "gpt-4o"
    try:
        encoding_name_for_model(model)
    except KeyError:
        # GPT-named model newer than the pinned tiktoken's mapping table
        # (e.g. gpt-5.6-*) — estimates must degrade to the default
        # encoding, not crash the LLM call.
        return "gpt-4o"
    return model


def _extract_tool_call_ids_from_message(msg: dict) -> set[str]:
    """
    Extract tool_call IDs from an assistant message.

    Supports all formats:
    - OpenAI Chat Completions: {"role": "assistant", "tool_calls": [{"id": "..."}]}
    - Anthropic: {"role": "assistant", "content": [{"type": "tool_use", "id": "..."}]}
    - OpenAI Responses API: {"type": "function_call", "call_id": "..."}

    Returns:
        Set of tool_call IDs found in the message.
    """
    ids: set[str] = set()

    # Responses API: standalone function_call item
    if msg.get("type") == "function_call":
        if call_id := msg.get("call_id"):
            ids.add(call_id)
        return ids

    if msg.get("role") != "assistant":
        return ids

    # OpenAI Chat Completions format: tool_calls array
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            tc_id = tc.get("id")
            if tc_id:
                ids.add(tc_id)

    # Anthropic format: content list with tool_use blocks
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tc_id = block.get("id")
                if tc_id:
                    ids.add(tc_id)

    return ids


def _extract_tool_response_ids_from_message(msg: dict) -> set[str]:
    """
    Extract tool_call IDs that this message is responding to.

    Supports all formats:
    - OpenAI Chat Completions: {"role": "tool", "tool_call_id": "..."}
    - Anthropic: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "..."}]}
    - OpenAI Responses API: {"type": "function_call_output", "call_id": "..."}

    Returns:
        Set of tool_call IDs this message responds to.
    """
    ids: set[str] = set()

    # Responses API: standalone function_call_output item
    if msg.get("type") == "function_call_output":
        if call_id := msg.get("call_id"):
            ids.add(call_id)
        return ids

    # OpenAI Chat Completions format: role=tool with tool_call_id
    if msg.get("role") == "tool":
        tc_id = msg.get("tool_call_id")
        if tc_id:
            ids.add(tc_id)

    # Anthropic format: content list with tool_result blocks
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tc_id = block.get("tool_use_id")
                if tc_id:
                    ids.add(tc_id)

    return ids


def _is_tool_response_message(msg: dict) -> bool:
    """Check if message is a tool response (Chat Completions, Anthropic, or Responses API)."""
    # Responses API format
    if msg.get("type") == "function_call_output":
        return True
    # OpenAI Chat Completions format
    if msg.get("role") == "tool":
        return True
    # Anthropic format
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
    return False


def _remove_orphan_tool_responses(
    messages: list[dict], orphan_ids: set[str]
) -> list[dict]:
    """
    Remove tool response messages/blocks that reference orphan tool_call IDs.

    Supports OpenAI Chat Completions, Anthropic, and Responses API formats.
    For Anthropic messages with mixed valid/orphan tool_result blocks,
    filters out only the orphan blocks instead of dropping the entire message.
    """
    result = []
    for msg in messages:
        # Responses API: function_call_output - drop if orphan
        if msg.get("type") == "function_call_output":
            if msg.get("call_id") in orphan_ids:
                continue
            result.append(msg)
            continue

        # OpenAI Chat Completions: role=tool - drop entire message if orphan
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id in orphan_ids:
                continue
            result.append(msg)
            continue

        # Anthropic format: content list may have mixed tool_result blocks
        content = msg.get("content")
        if isinstance(content, list):
            has_tool_results = any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            )
            if has_tool_results:
                # Filter out orphan tool_result blocks, keep valid ones
                filtered_content = [
                    block
                    for block in content
                    if not (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and block.get("tool_use_id") in orphan_ids
                    )
                ]
                # Only keep message if it has remaining content
                if filtered_content:
                    msg = msg.copy()
                    msg["content"] = filtered_content
                    result.append(msg)
                continue

        result.append(msg)
    return result


def validate_and_remove_orphan_tool_responses(
    messages: list[dict],
    log_warning: bool = True,
) -> list[dict]:
    """
    Validate tool_call/tool_response pairs and remove orphaned responses.

    Scans messages in order, tracking all tool_call IDs. Any tool response
    referencing an ID not seen in a preceding message is considered orphaned
    and removed. This prevents API errors like Anthropic's "unexpected tool_use_id".

    Args:
        messages: List of messages to validate (OpenAI or Anthropic format)
        log_warning: Whether to log a warning when orphans are found

    Returns:
        A new list with orphaned tool responses removed
    """
    available_ids: set[str] = set()
    orphan_ids: set[str] = set()

    for msg in messages:
        available_ids |= _extract_tool_call_ids_from_message(msg)
        for resp_id in _extract_tool_response_ids_from_message(msg):
            if resp_id not in available_ids:
                orphan_ids.add(resp_id)

    if not orphan_ids:
        return messages

    if log_warning:
        logger.warning(
            f"Removing {len(orphan_ids)} orphan tool response(s): {orphan_ids}"
        )

    return _remove_orphan_tool_responses(messages, orphan_ids)


def _ensure_tool_pairs_intact(
    recent_messages: list[dict],
    all_messages: list[dict],
    start_index: int,
) -> list[dict]:
    """
    Ensure tool_call/tool_response pairs stay together after slicing.

    When slicing messages for context compaction, a naive slice can separate
    an assistant message containing tool_calls from its corresponding tool
    response messages. This causes API validation errors (e.g., Anthropic's
    "unexpected tool_use_id found in tool_result blocks").

    This function checks for orphan tool responses in the slice and extends
    backwards to include their corresponding assistant messages.

    Supports both formats:
    - OpenAI: tool_calls array + role="tool" responses
    - Anthropic: tool_use blocks + tool_result blocks

    Args:
        recent_messages: The sliced messages to validate
        all_messages: The complete message list (for looking up missing assistants)
        start_index: The index in all_messages where recent_messages begins

    Returns:
        A potentially extended list of messages with tool pairs intact
    """
    if not recent_messages:
        return recent_messages

    # Collect all tool_call_ids from assistant messages in the slice
    available_tool_call_ids: set[str] = set()
    for msg in recent_messages:
        available_tool_call_ids |= _extract_tool_call_ids_from_message(msg)

    # Find orphan tool responses (responses whose tool_call_id is missing)
    orphan_tool_call_ids: set[str] = set()
    for msg in recent_messages:
        response_ids = _extract_tool_response_ids_from_message(msg)
        for tc_id in response_ids:
            if tc_id not in available_tool_call_ids:
                orphan_tool_call_ids.add(tc_id)

    if not orphan_tool_call_ids:
        # No orphans, slice is valid
        return recent_messages

    # Find the assistant messages that contain the orphan tool_call_ids
    # Search backwards from start_index in all_messages
    messages_to_prepend: list[dict] = []
    for i in range(start_index - 1, -1, -1):
        msg = all_messages[i]
        msg_tool_ids = _extract_tool_call_ids_from_message(msg)
        if msg_tool_ids & orphan_tool_call_ids:
            # This assistant message has tool_calls we need
            # Also collect its contiguous tool responses that follow it
            assistant_and_responses: list[dict] = [msg]

            # Scan forward from this assistant to collect tool responses
            for j in range(i + 1, start_index):
                following_msg = all_messages[j]
                following_response_ids = _extract_tool_response_ids_from_message(
                    following_msg
                )
                if following_response_ids and following_response_ids & msg_tool_ids:
                    assistant_and_responses.append(following_msg)
                elif not _is_tool_response_message(following_msg):
                    # Stop at first non-tool-response message
                    break

            # Prepend the assistant and its tool responses (maintain order)
            messages_to_prepend = assistant_and_responses + messages_to_prepend
            # Mark these as found
            orphan_tool_call_ids -= msg_tool_ids
            # Also add this assistant's tool_call_ids to available set
            available_tool_call_ids |= msg_tool_ids

        if not orphan_tool_call_ids:
            # Found all missing assistants
            break

    if orphan_tool_call_ids:
        # Some tool_call_ids couldn't be resolved - remove those tool responses
        # This shouldn't happen in normal operation but handles edge cases
        logger.warning(
            f"Could not find assistant messages for tool_call_ids: {orphan_tool_call_ids}. "
            "Removing orphan tool responses."
        )
        recent_messages = _remove_orphan_tool_responses(
            recent_messages, orphan_tool_call_ids
        )

    if messages_to_prepend:
        logger.info(
            f"Extended recent messages by {len(messages_to_prepend)} to preserve "
            f"tool_call/tool_response pairs"
        )
        return messages_to_prepend + recent_messages

    return recent_messages


# ---------------------------------------------------------------------------#
#  LLM SUMMARISATION — chunked map-reduce over the *whole* older history      #
# ---------------------------------------------------------------------------#

# Characters of flattened conversation handed to one summariser call.  The
# previous implementation truncated the entire older history to this many
# characters and silently dropped the rest; the history is now chunked at
# this size and every chunk is summarised, then the parts are merged.
SUMMARY_CHUNK_CHARS = 100_000

# Upper bound on chunks per compaction (32 × 100K chars ≈ 800K tokens).  Past
# it the *oldest* chunks are dropped and ``summary_coverage`` reports the loss
# instead of hiding it.
MAX_SUMMARY_CHUNKS = 32

_SUMMARY_CONCURRENCY = 6
_SUMMARY_MAX_TOKENS = 2_000
_MERGE_MAX_TOKENS = 3_000

SUMMARY_PREFIX = "[Previous conversation summary — for context only]: "

_SUMMARY_SYSTEM_PROMPT = (
    "Create a factual summary of the conversation so far. "
    "This summary will be used as context when continuing the conversation.\n\n"
    "CRITICAL: Only include information that is EXPLICITLY present in the "
    "conversation. Do NOT fabricate, infer, or invent any details. "
    "If a section has no relevant content in the conversation, skip it entirely.\n\n"
    "Before writing the summary, analyze each message chronologically to identify:\n"
    "- User requests and their explicit goals\n"
    "- Actions taken and key decisions made\n"
    "- Technical specifics (file names, tool outputs, function signatures)\n"
    "- Errors encountered and resolutions applied\n\n"
    "IMPORTANT: Preserve all concrete references verbatim — these are small but "
    "critical for continuing the conversation:\n"
    "- File paths and directory paths (e.g. /src/app/page.tsx, ./output/result.csv)\n"
    "- Image/media file paths from tool outputs\n"
    "- URLs, API endpoints, and webhook addresses\n"
    "- Resource IDs, session IDs, and identifiers\n"
    "- Tool names that were called and their key parameters\n"
    "- Environment variables, config keys, and credentials names (not values)\n\n"
    "Include ONLY the sections below that have relevant content "
    "(skip sections with nothing to report):\n\n"
    "## 1. Primary Request and Intent\n"
    "The user's explicit goals and what they are trying to accomplish.\n\n"
    "## 2. Key Technical Concepts\n"
    "Technologies, frameworks, tools, and patterns being used or discussed.\n\n"
    "## 3. Files and Resources Involved\n"
    "Specific files examined or modified, with relevant snippets and identifiers. "
    "Include exact file paths, image paths from tool outputs, and resource URLs.\n\n"
    "## 4. Artifact Trail\n"
    "Every file, agent, or resource that was CREATED, MODIFIED, or DELETED, "
    "with the tool that did it and the key change — this is what a "
    "continuation most often needs and most often loses.\n\n"
    "## 5. Errors and Fixes\n"
    "Problems encountered, error messages, and their resolutions.\n\n"
    "## 6. All User Messages\n"
    "A complete list of all user inputs (excluding tool outputs) "
    "to preserve their exact requests.\n\n"
    "## 7. Pending Tasks\n"
    "Work items the user explicitly requested that have not yet been completed.\n\n"
    "## 8. Current State\n"
    "What was happening most recently in the conversation."
)

_MERGE_SYSTEM_PROMPT = (
    "You are merging partial summaries of ONE conversation into a single "
    "summary. The parts are given in chronological order and were written "
    "from consecutive slices of the same conversation.\n\n"
    "Produce one summary with the same section structure as the parts. "
    "Preserve every concrete reference verbatim (file paths, URLs, IDs, tool "
    "names, identifiers). Where parts repeat a fact, state it once. Where "
    "parts disagree about state, the later part reflects the newer state. "
    "Do NOT add anything that is not present in the parts."
)


def _flatten_for_summary(msg: dict) -> str:
    """One message as the text the summariser reads.

    Covers Responses-API items, Chat Completions ``tool_calls`` (which the
    previous flattening skipped, so the summary never saw what tools the
    assistant had called) and list content, which is serialised whole so
    ``tool_use`` / ``tool_result`` blocks are visible to the summariser.
    """
    if msg.get("type") == "function_call":
        return (
            f"TOOL CALL ({msg.get('name', 'unknown_tool')}): {msg.get('arguments', '')}"
        )
    if msg.get("type") == "function_call_output":
        return f"TOOL OUTPUT: {msg.get('output', '')}"
    parts: list[str] = []
    role = str(msg.get("role", ""))
    content = msg.get("content", "")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, separators=(",", ":"))
    if content and role in ("user", "assistant", "tool", "system"):
        parts.append(f"{role.upper()}: {content}")
    for call in msg.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        parts.append(
            f"TOOL CALL ({fn.get('name', 'unknown_tool')}): {fn.get('arguments', '')}"
        )
    return "\n".join(parts)


def _chunk_texts(texts: list[str], max_chars: int) -> list[str]:
    """Greedily pack whole messages into chunks of at most *max_chars*.

    A single message longer than a chunk is split into pieces so nothing is
    skipped; message boundaries are otherwise respected.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if not text:
            continue
        if len(text) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            for i in range(0, len(text), max_chars):
                chunks.append(text[i : i + max_chars])
            continue
        if size + len(text) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(text)
        size += len(text) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


async def _summarize_chunk(
    client: AsyncOpenAI, model: str, text: str, part: int, total: int, timeout: float
) -> str:
    system = _SUMMARY_SYSTEM_PROMPT
    if total > 1:
        system += (
            f"\n\nThis is part {part} of {total} of the conversation, in "
            "chronological order. Summarise this part only; the parts will be "
            "merged afterwards."
        )
    response = await client.with_options(timeout=timeout).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Summarize:\n\n{text}"},
        ],
        max_tokens=_SUMMARY_MAX_TOKENS,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


async def _merge_summaries(
    client: AsyncOpenAI, model: str, parts: list[str], timeout: float
) -> str:
    joined = "\n\n".join(
        f"=== Part {i} of {len(parts)} ===\n{p}" for i, p in enumerate(parts, 1)
    )
    response = await client.with_options(timeout=timeout).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Merge these parts:\n\n{joined}"},
        ],
        max_tokens=_MERGE_MAX_TOKENS,
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


async def summarize_messages(
    messages: list[dict],
    client: AsyncOpenAI,
    model: str,
    timeout: float = 30.0,
) -> tuple[str, float]:
    """Summarise *messages* in full and report how much of them was read.

    Returns ``(summary_text, coverage)`` where *coverage* is the fraction of
    the flattened history the summariser actually saw — 1.0 unless the
    history exceeded ``MAX_SUMMARY_CHUNKS`` chunks, in which case the oldest
    chunks were dropped.
    """
    texts = [_flatten_for_summary(m) for m in messages]
    texts = [t for t in texts if t]
    total_chars = sum(len(t) for t in texts)
    if total_chars == 0:
        return "No conversation history available.", 1.0
    chunks = _chunk_texts(texts, SUMMARY_CHUNK_CHARS)
    coverage = 1.0
    if len(chunks) > MAX_SUMMARY_CHUNKS:
        dropped = chunks[: len(chunks) - MAX_SUMMARY_CHUNKS]
        chunks = chunks[len(chunks) - MAX_SUMMARY_CHUNKS :]
        coverage = 1.0 - sum(len(c) for c in dropped) / total_chars
        logger.warning(
            "Summariser input exceeds %d chunks; dropping the oldest %d "
            "(coverage %.0f%%)",
            MAX_SUMMARY_CHUNKS,
            len(dropped),
            coverage * 100,
        )
    if len(chunks) == 1:
        return await _summarize_chunk(client, model, chunks[0], 1, 1, timeout), coverage

    semaphore = asyncio.Semaphore(_SUMMARY_CONCURRENCY)

    async def one(index: int, chunk: str) -> str:
        async with semaphore:
            return await _summarize_chunk(
                client, model, chunk, index + 1, len(chunks), timeout
            )

    parts = await asyncio.gather(*(one(i, c) for i, c in enumerate(chunks)))
    merged = await _merge_summaries(client, model, list(parts), timeout)
    return merged, coverage


async def _summarize_messages_llm(
    messages: list[dict],
    client: AsyncOpenAI,
    model: str,
    timeout: float = 30.0,
) -> str:
    """Summary text only; see :func:`summarize_messages` for coverage."""
    text, _ = await summarize_messages(messages, client, model, timeout)
    return text


# ---------------------------------------------------------------------------#
#  Public token helpers (transcript.py trims the preserved final turn)        #
# ---------------------------------------------------------------------------#


def token_len(text: str, model: str) -> int:
    """Estimated tokens of *text* for *model*, in the same corrected space
    ``compress_context`` budgets in."""
    enc = encoding_for_model(_normalize_model_for_tokenizer(model))
    return int(_tok_len(text, enc) * _token_estimate_factor(model))


def truncate_middle(text: str, model: str, max_tokens: int) -> str:
    """Middle-out truncation of *text* to about *max_tokens* corrected tokens."""
    enc = encoding_for_model(_normalize_model_for_tokenizer(model))
    factor = _token_estimate_factor(model)
    return _truncate_middle_tokens(text, enc, max(1, int(max_tokens / factor)))


# ---------------------------------------------------------------------------#
#  compress_context                                                           #
# ---------------------------------------------------------------------------#


# Room left beside the verbatim tail for the summary of everything older.
_SUMMARY_ALLOWANCE_TOKENS = 6_000


def _tail_by_tokens(body: list[dict], budget: int, size) -> list[dict]:
    """The longest suffix of *body* within *budget* tokens.

    Always includes the last message, whatever its size.  If the message just
    before the chosen suffix is itself larger than *budget* — a generated
    document, a giant tool result — it is included too: truncated to fit it
    keeps most of its head and tail verbatim, whereas left to the summariser
    it would come back as a few lines.
    """
    tail: list[dict] = []
    used = 0
    for m in reversed(body):
        s = size(m)
        if tail and used + s > budget:
            break
        tail.append(m)
        used += s
    start = len(body) - len(tail)
    if start > 0 and size(body[start - 1]) > budget:
        tail.append(body[start - 1])
    tail.reverse()
    return tail


def _cuttable_parts(msg: dict, truncate_tool_arguments: bool) -> list[tuple]:
    """(getter, setter) pairs for every string in *msg* that truncation may
    shorten.  Tool-call arguments are included only when the caller renders
    them as text — a middle-out cut breaks their JSON for callers that replay
    them to an API."""
    parts: list[tuple] = []
    if msg.get("type") == "function_call_output":
        if isinstance(msg.get("output"), str):
            parts.append(
                (lambda m=msg: m["output"], lambda v, m=msg: m.__setitem__("output", v))
            )
        return parts
    if msg.get("type") == "function_call":
        if truncate_tool_arguments and isinstance(msg.get("arguments"), str):
            parts.append(
                (
                    lambda m=msg: m["arguments"],
                    lambda v, m=msg: m.__setitem__("arguments", v),
                )
            )
        return parts
    content = msg.get("content")
    if isinstance(content, str) and content:
        parts.append(
            (lambda m=msg: m["content"], lambda v, m=msg: m.__setitem__("content", v))
        )
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_result" and isinstance(
                item.get("content"), str
            ):
                parts.append(
                    (
                        lambda it=item: it["content"],
                        lambda v, it=item: it.__setitem__("content", v),
                    )
                )
            elif item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(
                    (
                        lambda it=item: it["text"],
                        lambda v, it=item: it.__setitem__("text", v),
                    )
                )
    if truncate_tool_arguments:
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                parts.append(
                    (
                        lambda f=fn: f["arguments"],
                        lambda v, f=fn: f.__setitem__("arguments", v),
                    )
                )
    return parts


def _truncate_to_fit(
    msgs: list[dict],
    budget: int,
    enc,
    factor: float,
    protected: set[int],
    floor_cap: int,
    start_cap: int | None,
    truncate_tool_arguments: bool,
    size,
) -> int:
    """Shorten the largest messages just enough to fit *budget*.

    Water-filling: find the largest per-message cap ``C`` such that every
    cuttable message shortened to ``C`` fits, then apply it.  Only messages
    above the cap lose anything, and they keep as much as the budget allows
    — unlike a fixed cap, which cut a 120K-token response to 8K when 75K
    would have fit.  *start_cap*, when a caller passes one, bounds ``C``
    from above (legacy behaviour); *floor_cap* bounds it from below.
    """
    cuttable: list[tuple[int, list[tuple], int]] = []  # (index, parts, tokens)
    for i, m in enumerate(msgs):
        if id(m) in protected:
            continue
        parts = _cuttable_parts(m, truncate_tool_arguments)
        if not parts:
            continue
        tokens = sum(_tok_len(get(), enc) for get, _ in parts)
        if tokens > 0:
            cuttable.append((i, parts, tokens))
    if not cuttable:
        return sum(size(m) for m in msgs)

    total_unscaled = sum(_msg_tokens(m, enc) for m in msgs)
    fixed = total_unscaled - sum(t for _, _, t in cuttable)
    target_unscaled = budget / factor

    def fits(cap: int) -> bool:
        return fixed + sum(min(t, cap) for _, _, t in cuttable) <= target_unscaled

    lo, hi = floor_cap, max(t for _, _, t in cuttable)
    if start_cap is not None:
        hi = min(hi, start_cap)
    cap = lo
    if hi >= lo and fits(hi):
        cap = hi
    elif hi > lo:
        # largest cap that fits, or the floor
        a, b = lo, hi
        while a < b:
            mid = (a + b + 1) // 2
            if fits(mid):
                a = mid
            else:
                b = mid - 1
        cap = a

    for _round in range(4):
        for _, parts, tokens in cuttable:
            if tokens <= cap:
                continue
            for get, put in parts:
                text = get()
                part_tokens = _tok_len(text, enc)
                share = max(1, int(cap * part_tokens / tokens))
                if part_tokens > share:
                    put(_truncate_middle_tokens(text, enc, share))
        total = sum(size(m) for m in msgs)
        if total <= budget or cap <= floor_cap:
            return total
        # Rounding and wrapper slack: tighten and go again.
        cap = max(floor_cap, int(cap * 0.9))
        cuttable = [
            (i, parts, sum(_tok_len(get(), enc) for get, _ in parts))
            for i, parts, _ in cuttable
        ]
    return sum(size(m) for m in msgs)


def _delete_oldest(
    msgs: list[dict], budget: int, protected: set[int], size, first_deletable: int
) -> tuple[int, int]:
    """Drop the oldest unprotected messages (never the last) until *budget*
    fits; a deleted tool call takes its responses with it.  Returns
    ``(total, dropped)``.  Sizes are computed once and adjusted per deletion
    — recounting the whole history after each one is quadratic tokenization.
    """
    sizes = [size(m) for m in msgs]
    total = sum(sizes)
    dropped = 0
    while total > budget:
        victim = next(
            (
                i
                for i in range(first_deletable, len(msgs) - 1)
                if id(msgs[i]) not in protected
            ),
            None,
        )
        if victim is None:
            break
        call_ids = _extract_tool_call_ids_from_message(msgs[victim])
        remove = {victim}
        if call_ids:
            for j in range(victim + 1, len(msgs) - 1):
                if _extract_tool_response_ids_from_message(msgs[j]) & call_ids:
                    remove.add(j)
        total -= sum(sizes[k] for k in remove)
        msgs[:] = [m for k, m in enumerate(msgs) if k not in remove]
        sizes = [sz for k, sz in enumerate(sizes) if k not in remove]
        dropped += len(remove)
    return total, dropped


async def compress_context(
    messages: list[dict],
    target_tokens: int | None = None,
    *,
    model: str = "gpt-4o",
    client: AsyncOpenAI | None = None,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    reserve: int = DEFAULT_COMPRESSION_RESERVE,
    start_cap: int | None = None,
    floor_cap: int = 128,
    keep_recent_tokens: int | None = None,
    truncate_tool_arguments: bool = False,
) -> CompressResult:
    """Fit *messages* under *target_tokens* with the least loss of what matters.

    Strategy, in order — each phase runs only if the previous left the
    history over budget:

    1. **Pick the verbatim tail; summarise everything older.**  With
       ``keep_recent_tokens`` the tail is the most recent history that fits
       the budget minus room for a summary (that value is the floor, not the
       ceiling); with ``keep_recent`` it is a message count, for callers
       that still pass one.  An oversized message just before the tail is
       pulled in so it is truncated rather than summarised away.  Tool
       pairs stay intact.  With a *client*, everything older is summarised
       *in full* — chunked map-reduce, see :func:`summarize_messages` —
       into one message, and ``summary_coverage`` on the result says how
       much of it the summariser read.
    2. **Shorten the largest older messages just enough.**  A single
       per-message cap is chosen so that the history fits, and only
       messages above it are middle-out truncated to it.  The system
       prompt, the summary and objective messages are never cut.  Tool-call
       *arguments* are cut only when ``truncate_tool_arguments`` is set
       (callers that render history as text); callers that replay tool
       calls to an API keep them intact.
    3. **Delete oldest first.**  Never the last message, never the tail; a
       deleted tool call takes its responses with it.
    4. **Then the tail, the same way**, and as a last resort every
       unprotected message down to ``floor_cap``.

    ``was_compacted`` is False when the history already fit.  ``error`` is
    set when even the last resort left it over budget.
    """
    if target_tokens is None:
        target_tokens = get_compression_target(model)
    if not messages:
        return CompressResult(
            messages=[], token_count=0, was_compacted=False, original_token_count=0
        )

    token_model = _normalize_model_for_tokenizer(model)
    enc = encoding_for_model(token_model)
    factor = _token_estimate_factor(model)
    msgs = deepcopy(messages)

    def size(m: dict) -> int:
        return int(_msg_tokens(m, enc) * factor)

    original_count = sum(size(m) for m in msgs)
    budget = target_tokens - reserve
    if original_count <= budget:
        return CompressResult(
            messages=msgs,
            token_count=original_count,
            was_compacted=False,
            original_token_count=original_count,
            summarizer_available=client is not None,
        )

    messages_summarized = 0
    messages_dropped = 0
    coverage = 1.0
    tail_tokens = 0
    protected: set[int] = set()

    has_system = bool(msgs) and msgs[0].get("role") == "system"
    head = msgs[:1] if has_system else []
    body = msgs[len(head) :]

    # ---- 1. pick the verbatim tail; summarise everything older ----------------
    # The tail is chosen whether or not a summariser is available: without
    # one it is what the truncation and deletion phases below leave alone
    # until nothing else is left to cut.  With a token budget the tail grows
    # to fill the target minus room for the summary — a restart wants as
    # much recent history verbatim as the budget allows, and
    # ``keep_recent_tokens`` is the floor, not the ceiling.
    tail: list[dict] = []
    if len(body) >= 2:
        if keep_recent_tokens is not None:
            tail_budget = max(keep_recent_tokens, budget - _SUMMARY_ALLOWANCE_TOKENS)
            tail = _tail_by_tokens(body, tail_budget, size)
        else:
            tail = body[-keep_recent:] if len(body) > keep_recent else list(body)
        start = len(body) - len(tail)
        tail = _ensure_tool_pairs_intact(tail, body, start)
    tail_ids = {id(m) for m in tail}
    if client is not None and tail and len(tail) < len(body):
        start = len(body) - len(tail)
        old = body[:start]
        if old:
            try:
                summary_text, coverage = await summarize_messages(old, client, model)
                summary_msg = {
                    "role": "assistant",
                    "content": SUMMARY_PREFIX + summary_text,
                }
                protected.add(id(summary_msg))
                msgs = head + [summary_msg] + tail
                messages_summarized = len(old)
                logger.info(
                    "Context summarized: %d -> %d tokens, summarized %d messages "
                    "(coverage %.0f%%)",
                    original_count,
                    sum(size(m) for m in msgs),
                    messages_summarized,
                    coverage * 100,
                )
            except Exception as e:
                logger.warning(
                    "Summarization failed, continuing with truncation: %s", e
                )
    tail_tokens = sum(size(m) for m in tail)

    for m in head:
        protected.add(id(m))
    for m in msgs:
        if _is_objective_message(m):
            protected.add(id(m))

    # ---- 2. structured content of non-tool messages becomes text -------------
    for m in msgs:
        c = m.get("content")
        if c is not None and not isinstance(c, str) and not _is_tool_message(m):
            m["content"] = json.dumps(c, separators=(",", ":"))

    total = sum(size(m) for m in msgs)

    # ---- 3. shorten the largest OLD messages just enough ----------------------
    # The tail is spared here: it is the recent history the caller asked to
    # keep, and older content should give way first.
    if total > budget:
        total = _truncate_to_fit(
            msgs,
            budget,
            enc,
            factor,
            protected | tail_ids,
            floor_cap,
            start_cap,
            truncate_tool_arguments,
            size,
        )

    # ---- 4. delete oldest first (never the last, never the tail) --------------
    if total > budget:
        first_deletable = len(head) + (1 if messages_summarized else 0)
        total, messages_dropped = _delete_oldest(
            msgs, budget, protected | tail_ids, size, first_deletable
        )

    # ---- 5. now the tail too: shorten its largest just enough -----------------
    if total > budget:
        total = _truncate_to_fit(
            msgs,
            budget,
            enc,
            factor,
            protected,
            floor_cap,
            start_cap,
            truncate_tool_arguments,
            size,
        )

    # ---- 6. floor everything unprotected ---------------------------------------
    if total > budget:
        total = _truncate_to_fit(
            msgs,
            budget,
            enc,
            factor,
            protected,
            floor_cap,
            floor_cap,
            truncate_tool_arguments,
            size,
        )

    final_msgs = validate_and_remove_orphan_tool_responses(
        [m for m in msgs if m is not None]
    )
    final_count = sum(size(m) for m in final_msgs)
    error = None
    if final_count > budget:
        error = f"Could not compress below target ({final_count + reserve} > {target_tokens})"
        logger.warning(error)

    return CompressResult(
        messages=final_msgs,
        token_count=final_count,
        was_compacted=True,
        error=error,
        original_token_count=original_count,
        messages_summarized=messages_summarized,
        messages_dropped=messages_dropped,
        summary_coverage=coverage,
        summarizer_available=client is not None,
        tail_tokens=tail_tokens,
    )
