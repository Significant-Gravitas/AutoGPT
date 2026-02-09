from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tiktoken import encoding_for_model

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
    """
    WRAPPER = 3 + (1 if "name" in msg else 0)

    # Count content tokens
    content_tokens = _tok_len(msg.get("content") or "", enc)

    # Count tool call tokens for both OpenAI and Anthropic formats
    tool_call_tokens = 0

    # OpenAI format: tool_calls array at message level
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
    content = msg.get("content")
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
            elif isinstance(item, dict) and "content" in item:
                # Other content types with content field
                tool_call_tokens += _tok_len(item.get("content", ""), enc)
        # For list content, override content_tokens since we counted everything above
        content_tokens = 0

    return WRAPPER + content_tokens + tool_call_tokens


def _is_tool_message(msg: dict) -> bool:
    """Check if a message contains tool calls or results that should be protected."""
    content = msg.get("content")

    # Check for Anthropic-style tool messages
    if isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type") in ("tool_use", "tool_result")
        for item in content
    ):
        return True

    # Check for OpenAI-style tool calls in the message
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
    Handles both Anthropic-style (list content) and OpenAI-style (string content) tool messages.
    """
    content = msg.get("content")

    # OpenAI-style tool message: role="tool" with string content
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

    # Split the allowance between the two ends:
    head = max_tok // 2 - 1  # -1 for the ellipsis
    tail = max_tok - head - 1
    mid = enc.encode(" … ")
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
    return sum(_msg_tokens(m, enc) for m in messages)


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
    return _tok_len(text, enc)


# ---------------------------------------------------------------------------#
#  UNIFIED CONTEXT COMPRESSION                                               #
# ---------------------------------------------------------------------------#

# Default thresholds
DEFAULT_TOKEN_THRESHOLD = 120_000
DEFAULT_KEEP_RECENT = 15


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


def _normalize_model_for_tokenizer(model: str) -> str:
    """Normalize model name for tiktoken tokenizer selection."""
    if "/" in model:
        model = model.split("/")[-1]
    if "claude" in model.lower() or not any(
        known in model.lower() for known in ["gpt", "o1", "chatgpt", "text-"]
    ):
        return "gpt-4o"
    return model


def _extract_tool_call_ids_from_message(msg: dict) -> set[str]:
    """
    Extract tool_call IDs from an assistant message.

    Supports both formats:
    - OpenAI: {"role": "assistant", "tool_calls": [{"id": "..."}]}
    - Anthropic: {"role": "assistant", "content": [{"type": "tool_use", "id": "..."}]}

    Returns:
        Set of tool_call IDs found in the message.
    """
    ids: set[str] = set()
    if msg.get("role") != "assistant":
        return ids

    # OpenAI format: tool_calls array
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

    Supports both formats:
    - OpenAI: {"role": "tool", "tool_call_id": "..."}
    - Anthropic: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "..."}]}

    Returns:
        Set of tool_call IDs this message responds to.
    """
    ids: set[str] = set()

    # OpenAI format: role=tool with tool_call_id
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
    """Check if message is a tool response (OpenAI or Anthropic format)."""
    # OpenAI format
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

    Supports both OpenAI and Anthropic formats.
    For Anthropic messages with mixed valid/orphan tool_result blocks,
    filters out only the orphan blocks instead of dropping the entire message.
    """
    result = []
    for msg in messages:
        # OpenAI format: role=tool - drop entire message if orphan
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


async def _summarize_messages_llm(
    messages: list[dict],
    client: AsyncOpenAI,
    model: str,
    timeout: float = 30.0,
) -> str:
    """Summarize messages using an LLM."""
    conversation = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content and role in ("user", "assistant", "tool"):
            conversation.append(f"{role.upper()}: {content}")

    conversation_text = "\n\n".join(conversation)

    if not conversation_text:
        return "No conversation history available."

    # Limit to ~100k chars for safety
    MAX_CHARS = 100_000
    if len(conversation_text) > MAX_CHARS:
        conversation_text = conversation_text[:MAX_CHARS] + "\n\n[truncated]"

    response = await client.with_options(timeout=timeout).chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a detailed summary of the conversation so far. "
                    "This summary will be used as context when continuing the conversation.\n\n"
                    "Before writing the summary, analyze each message chronologically to identify:\n"
                    "- User requests and their explicit goals\n"
                    "- Your approach and key decisions made\n"
                    "- Technical specifics (file names, tool outputs, function signatures)\n"
                    "- Errors encountered and resolutions applied\n\n"
                    "You MUST include ALL of the following sections:\n\n"
                    "## 1. Primary Request and Intent\n"
                    "The user's explicit goals and what they are trying to accomplish.\n\n"
                    "## 2. Key Technical Concepts\n"
                    "Technologies, frameworks, tools, and patterns being used or discussed.\n\n"
                    "## 3. Files and Resources Involved\n"
                    "Specific files examined or modified, with relevant snippets and identifiers.\n\n"
                    "## 4. Errors and Fixes\n"
                    "Problems encountered, error messages, and their resolutions. "
                    "Include any user feedback on fixes.\n\n"
                    "## 5. Problem Solving\n"
                    "Issues that have been resolved and how they were addressed.\n\n"
                    "## 6. All User Messages\n"
                    "A complete list of all user inputs (excluding tool outputs) to preserve their exact requests.\n\n"
                    "## 7. Pending Tasks\n"
                    "Work items the user explicitly requested that have not yet been completed.\n\n"
                    "## 8. Current Work\n"
                    "Precise description of what was being worked on most recently, including relevant context.\n\n"
                    "## 9. Next Steps\n"
                    "What should happen next, aligned with the user's most recent requests. "
                    "Include verbatim quotes of recent instructions if relevant."
                ),
            },
            {"role": "user", "content": f"Summarize:\n\n{conversation_text}"},
        ],
        max_tokens=1500,
        temperature=0.3,
    )

    return response.choices[0].message.content or "No summary available."


async def compress_context(
    messages: list[dict],
    target_tokens: int = DEFAULT_TOKEN_THRESHOLD,
    *,
    model: str = "gpt-4o",
    client: AsyncOpenAI | None = None,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    reserve: int = 2_048,
    start_cap: int = 8_192,
    floor_cap: int = 128,
) -> CompressResult:
    """
    Unified context compression that combines summarization and truncation strategies.

    Strategy (in order):
    1. **LLM summarization** – If client provided, summarize old messages into a
       single context message while keeping recent messages intact. This is the
       primary strategy for chat service.
    2. **Content truncation** – Progressively halve a per-message cap and truncate
       bloated message content (tool outputs, large pastes). Preserves all messages
       but shortens their content. Primary strategy when client=None (LLM blocks).
    3. **Middle-out deletion** – Delete whole messages one at a time from the center
       outward, skipping tool messages and objective messages.
    4. **First/last trim** – Truncate first and last message content as last resort.

    Parameters
    ----------
    messages        Complete chat history (will be deep-copied).
    target_tokens   Hard ceiling for prompt size.
    model           Model name for tokenization and summarization.
    client          AsyncOpenAI client. If provided, enables LLM summarization
                    as the first strategy. If None, skips to truncation strategies.
    keep_recent     Number of recent messages to preserve during summarization.
    reserve         Tokens to reserve for model response.
    start_cap       Initial per-message truncation ceiling (tokens).
    floor_cap       Lowest cap before moving to deletions.

    Returns
    -------
    CompressResult with compressed messages and metadata.
    """
    # Guard clause for empty messages
    if not messages:
        return CompressResult(
            messages=[],
            token_count=0,
            was_compacted=False,
            original_token_count=0,
        )

    token_model = _normalize_model_for_tokenizer(model)
    enc = encoding_for_model(token_model)
    msgs = deepcopy(messages)

    def total_tokens() -> int:
        return sum(_msg_tokens(m, enc) for m in msgs)

    original_count = total_tokens()

    # Already under limit
    if original_count + reserve <= target_tokens:
        return CompressResult(
            messages=msgs,
            token_count=original_count,
            was_compacted=False,
            original_token_count=original_count,
        )

    messages_summarized = 0
    messages_dropped = 0

    # ---- STEP 1: LLM summarization (if client provided) -------------------
    # This is the primary compression strategy for chat service.
    # Summarize old messages while keeping recent ones intact.
    if client is not None:
        has_system = len(msgs) > 0 and msgs[0].get("role") == "system"
        system_msg = msgs[0] if has_system else None

        # Calculate old vs recent messages
        if has_system:
            if len(msgs) > keep_recent + 1:
                old_msgs = msgs[1:-keep_recent]
                recent_msgs = msgs[-keep_recent:]
            else:
                old_msgs = []
                recent_msgs = msgs[1:] if len(msgs) > 1 else []
        else:
            if len(msgs) > keep_recent:
                old_msgs = msgs[:-keep_recent]
                recent_msgs = msgs[-keep_recent:]
            else:
                old_msgs = []
                recent_msgs = msgs

        # Ensure tool pairs stay intact
        slice_start = max(0, len(msgs) - keep_recent)
        recent_msgs = _ensure_tool_pairs_intact(recent_msgs, msgs, slice_start)

        if old_msgs:
            try:
                summary_text = await _summarize_messages_llm(old_msgs, client, model)
                summary_msg = {
                    "role": "assistant",
                    "content": f"[Previous conversation summary — for context only]: {summary_text}",
                }
                messages_summarized = len(old_msgs)

                if has_system:
                    msgs = [system_msg, summary_msg] + recent_msgs
                else:
                    msgs = [summary_msg] + recent_msgs

                logger.info(
                    f"Context summarized: {original_count} -> {total_tokens()} tokens, "
                    f"summarized {messages_summarized} messages"
                )
            except Exception as e:
                logger.warning(f"Summarization failed, continuing with truncation: {e}")
                # Fall through to content truncation

    # ---- STEP 2: Normalize content ----------------------------------------
    # Convert non-string payloads to strings so token counting is coherent.
    # Always run this before truncation to ensure consistent token counting.
    for i, m in enumerate(msgs):
        if not isinstance(m.get("content"), str) and m.get("content") is not None:
            if _is_tool_message(m):
                continue
            if i == 0 or i == len(msgs) - 1:
                continue
            content_str = json.dumps(m["content"], separators=(",", ":"))
            if len(content_str) > 20_000:
                content_str = _truncate_middle_tokens(content_str, enc, 20_000)
            m["content"] = content_str

    # ---- STEP 3: Token-aware content truncation ---------------------------
    # Progressively halve per-message cap and truncate bloated content.
    # This preserves all messages but shortens their content.
    cap = start_cap
    while total_tokens() + reserve > target_tokens and cap >= floor_cap:
        for m in msgs[1:-1]:
            if _is_tool_message(m):
                _truncate_tool_message_content(m, enc, cap)
                continue
            if _is_objective_message(m):
                continue
            content = m.get("content") or ""
            if _tok_len(content, enc) > cap:
                m["content"] = _truncate_middle_tokens(content, enc, cap)
        cap //= 2

    # ---- STEP 4: Middle-out deletion --------------------------------------
    # Delete messages one at a time from the center outward.
    # This is more granular than dropping all old messages at once.
    while total_tokens() + reserve > target_tokens and len(msgs) > 2:
        deletable: list[int] = []
        for i in range(1, len(msgs) - 1):
            msg = msgs[i]
            if (
                msg is not None
                and not _is_tool_message(msg)
                and not _is_objective_message(msg)
            ):
                deletable.append(i)
        if not deletable:
            break
        centre = len(msgs) // 2
        to_delete = min(deletable, key=lambda i: abs(i - centre))
        del msgs[to_delete]
        messages_dropped += 1

    # ---- STEP 5: Final trim on first/last ---------------------------------
    cap = start_cap
    while total_tokens() + reserve > target_tokens and cap >= floor_cap:
        for idx in (0, -1):
            msg = msgs[idx]
            if msg is None:
                continue
            if _is_tool_message(msg):
                _truncate_tool_message_content(msg, enc, cap)
                continue
            text = msg.get("content") or ""
            if _tok_len(text, enc) > cap:
                msg["content"] = _truncate_middle_tokens(text, enc, cap)
        cap //= 2

    # Filter out any None values that may have been introduced
    final_msgs: list[dict] = [m for m in msgs if m is not None]
    final_count = sum(_msg_tokens(m, enc) for m in final_msgs)
    error = None
    if final_count + reserve > target_tokens:
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
    )
