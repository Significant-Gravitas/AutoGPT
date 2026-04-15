"""Tests for prompt.py compatibility with the OpenAI Responses API.

The Responses API uses a different conversation format:
- Tool calls are standalone items with ``type: "function_call"`` and ``call_id``
- Tool results are items with ``type: "function_call_output"`` and ``call_id``
- These items do NOT have ``role`` at the top level

These tests validate that prompt utilities correctly handle Responses API items
alongside Chat Completions and Anthropic formats.
"""

import pytest
from tiktoken import encoding_for_model

from backend.util.prompt import (
    _ensure_tool_pairs_intact,
    _extract_tool_call_ids_from_message,
    _extract_tool_response_ids_from_message,
    _is_tool_message,
    _is_tool_response_message,
    _msg_tokens,
    _remove_orphan_tool_responses,
    _truncate_tool_message_content,
    compress_context,
    validate_and_remove_orphan_tool_responses,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def enc():
    return encoding_for_model("gpt-4o")


# ── Sample items ──────────────────────────────────────────────────────────

FUNCTION_CALL_ITEM = {
    "type": "function_call",
    "id": "fc_abc",
    "call_id": "call_abc",
    "name": "search_tool",
    "arguments": '{"query": "python asyncio tutorial"}',
    "status": "completed",
}

FUNCTION_CALL_OUTPUT_ITEM = {
    "type": "function_call_output",
    "call_id": "call_abc",
    "output": '{"results": ["result1", "result2", "result3"]}',
}


# ═══════════════════════════════════════════════════════════════════════════
# _msg_tokens
# ═══════════════════════════════════════════════════════════════════════════


class TestMsgTokensResponsesApi:
    """_msg_tokens should count tokens in function_call / function_call_output
    items, not just role-based messages."""

    def test_chat_completions_tool_call_counted(self, enc):
        """Baseline: Chat Completions tool_calls are counted correctly."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "search_tool",
                        "arguments": '{"query": "python asyncio tutorial"}',
                    },
                }
            ],
        }
        tokens = _msg_tokens(msg, enc)
        assert tokens > 10  # Should count the tool call content

    def test_chat_completions_tool_response_counted(self, enc):
        """Baseline: Chat Completions tool responses are counted correctly."""
        msg = {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": '{"results": ["result1", "result2"]}',
        }
        tokens = _msg_tokens(msg, enc)
        assert tokens > 5

    def test_function_call_minimal_fields(self, enc):
        """function_call with missing optional fields still counts."""
        msg = {"type": "function_call"}
        tokens = _msg_tokens(msg, enc)
        assert tokens >= 3  # At least the wrapper

    def test_function_call_output_minimal_fields(self, enc):
        """function_call_output with missing output field still counts."""
        msg = {"type": "function_call_output"}
        tokens = _msg_tokens(msg, enc)
        assert tokens >= 3

    def test_function_call_arguments_counted(self, enc):
        """function_call items have 'arguments' not 'content' — tokens must
        include the arguments string and the function name."""
        tokens = _msg_tokens(FUNCTION_CALL_ITEM, enc)
        # Must count at least the arguments and name tokens
        name_tokens = len(enc.encode(FUNCTION_CALL_ITEM["name"]))
        args_tokens = len(enc.encode(FUNCTION_CALL_ITEM["arguments"]))
        assert tokens >= name_tokens + args_tokens

    def test_function_call_output_content_counted(self, enc):
        """function_call_output items have 'output' not 'content' — tokens must
        include the output string."""
        tokens = _msg_tokens(FUNCTION_CALL_OUTPUT_ITEM, enc)
        output_tokens = len(enc.encode(FUNCTION_CALL_OUTPUT_ITEM["output"]))
        assert tokens >= output_tokens


# ═══════════════════════════════════════════════════════════════════════════
# _is_tool_message
# ═══════════════════════════════════════════════════════════════════════════


class TestIsToolMessageResponsesApi:
    """_is_tool_message should recognise Responses API items as tool messages
    so they are protected from deletion during compaction."""

    def test_chat_completions_tool_call_detected(self):
        """Baseline: Chat Completions tool_calls are detected."""
        msg = {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function"}],
        }
        assert _is_tool_message(msg) is True

    def test_chat_completions_tool_response_detected(self):
        """Baseline: Chat Completions role=tool is detected."""
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "result"}
        assert _is_tool_message(msg) is True

    def test_anthropic_tool_use_detected(self):
        """Baseline: Anthropic tool_use is detected."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "t", "input": {}}
            ],
        }
        assert _is_tool_message(msg) is True

    def test_anthropic_tool_result_detected(self):
        """Baseline: Anthropic tool_result is detected."""
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
            ],
        }
        assert _is_tool_message(msg) is True

    def test_function_call_detected(self):
        """type=function_call should be recognised as a tool message."""
        assert _is_tool_message(FUNCTION_CALL_ITEM) is True

    def test_function_call_output_detected(self):
        """type=function_call_output should be recognised as a tool message."""
        assert _is_tool_message(FUNCTION_CALL_OUTPUT_ITEM) is True

    def test_regular_user_message_not_tool(self):
        """Plain user message → not a tool message."""
        assert _is_tool_message({"role": "user", "content": "hello"}) is False

    def test_regular_assistant_message_not_tool(self):
        """Plain assistant message without tool_calls → not a tool message."""
        assert _is_tool_message({"role": "assistant", "content": "hi"}) is False


# ═══════════════════════════════════════════════════════════════════════════
# _extract_tool_call_ids_from_message
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractToolCallIdsResponsesApi:
    """_extract_tool_call_ids_from_message should extract call_ids from
    Responses API function_call items."""

    def test_chat_completions_extracted(self):
        """Baseline: Chat Completions tool_calls IDs are extracted."""
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function"},
                {"id": "call_2", "type": "function"},
            ],
        }
        assert _extract_tool_call_ids_from_message(msg) == {"call_1", "call_2"}

    def test_anthropic_extracted(self):
        """Baseline: Anthropic tool_use IDs are extracted."""
        msg = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1"}],
        }
        assert _extract_tool_call_ids_from_message(msg) == {"toolu_1"}

    def test_function_call_extracted(self):
        """type=function_call with call_id should be extracted."""
        assert _extract_tool_call_ids_from_message(FUNCTION_CALL_ITEM) == {"call_abc"}

    def test_function_call_missing_call_id(self):
        """function_call without call_id → empty set."""
        msg = {"type": "function_call", "name": "tool"}
        assert _extract_tool_call_ids_from_message(msg) == set()

    def test_non_assistant_non_function_call(self):
        """Messages with neither role=assistant nor type=function_call → empty."""
        msg = {"role": "user", "content": "hello"}
        assert _extract_tool_call_ids_from_message(msg) == set()


# ═══════════════════════════════════════════════════════════════════════════
# _extract_tool_response_ids_from_message
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractToolResponseIdsResponsesApi:
    """_extract_tool_response_ids_from_message should extract call_ids from
    Responses API function_call_output items."""

    def test_chat_completions_extracted(self):
        """Baseline: Chat Completions tool_call_id is extracted."""
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "result"}
        assert _extract_tool_response_ids_from_message(msg) == {"call_1"}

    def test_anthropic_extracted(self):
        """Baseline: Anthropic tool_use_id is extracted."""
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
            ],
        }
        assert _extract_tool_response_ids_from_message(msg) == {"toolu_1"}

    def test_function_call_output_extracted(self):
        """type=function_call_output with call_id should be extracted."""
        assert _extract_tool_response_ids_from_message(FUNCTION_CALL_OUTPUT_ITEM) == {
            "call_abc"
        }

    def test_function_call_output_missing_call_id(self):
        """function_call_output without call_id → empty set."""
        msg = {"type": "function_call_output", "output": "result"}
        assert _extract_tool_response_ids_from_message(msg) == set()

    def test_non_tool_non_function_call_output(self):
        """Regular user message → empty set."""
        msg = {"role": "user", "content": "hello"}
        assert _extract_tool_response_ids_from_message(msg) == set()


# ═══════════════════════════════════════════════════════════════════════════
# _is_tool_response_message
# ═══════════════════════════════════════════════════════════════════════════


class TestIsToolResponseMessageResponsesApi:
    def test_chat_completions_detected(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "r"}
        assert _is_tool_response_message(msg) is True

    def test_anthropic_detected(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
            ],
        }
        assert _is_tool_response_message(msg) is True

    def test_function_call_output_detected(self):
        """type=function_call_output should be recognised as a tool response."""
        assert _is_tool_response_message(FUNCTION_CALL_OUTPUT_ITEM) is True

    def test_function_call_is_not_response(self):
        """function_call is a tool REQUEST, not a response."""
        assert _is_tool_response_message(FUNCTION_CALL_ITEM) is False

    def test_regular_message_not_response(self):
        """Plain message → not a tool response."""
        assert _is_tool_response_message({"role": "user", "content": "hi"}) is False


# ═══════════════════════════════════════════════════════════════════════════
# _truncate_tool_message_content
# ═══════════════════════════════════════════════════════════════════════════


class TestTruncateToolMessageContentResponsesApi:
    def test_chat_completions_truncated(self, enc):
        """Baseline: role=tool content is truncated."""
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "x" * 10000}
        _truncate_tool_message_content(msg, enc, max_tokens=50)
        assert len(enc.encode(msg["content"])) <= 55  # ~50 with rounding

    def test_function_call_output_truncated(self, enc):
        """function_call_output 'output' field should be truncated."""
        msg = {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "x" * 10000,
        }
        _truncate_tool_message_content(msg, enc, max_tokens=50)
        assert len(enc.encode(msg["output"])) <= 55

    def test_function_call_output_short_not_truncated(self, enc):
        """Short function_call_output output is left unchanged."""
        msg = {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "short",
        }
        _truncate_tool_message_content(msg, enc, max_tokens=1000)
        assert msg["output"] == "short"

    def test_function_call_not_truncated(self, enc):
        """function_call items (requests) should not be truncated."""
        msg = dict(FUNCTION_CALL_ITEM)  # copy
        original_args = msg["arguments"]
        _truncate_tool_message_content(msg, enc, max_tokens=5)
        assert msg["arguments"] == original_args  # unchanged


# ═══════════════════════════════════════════════════════════════════════════
# _remove_orphan_tool_responses
# ═══════════════════════════════════════════════════════════════════════════


class TestRemoveOrphanToolResponsesResponsesApi:
    def test_chat_completions_orphan_removed(self):
        """Baseline: orphan role=tool messages are removed."""
        messages = [
            {"role": "tool", "tool_call_id": "call_orphan", "content": "result"},
            {"role": "user", "content": "Hello"},
        ]
        result = _remove_orphan_tool_responses(messages, {"call_orphan"})
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_function_call_output_orphan_removed(self):
        """Orphan function_call_output items should be removed."""
        messages = [
            {
                "type": "function_call_output",
                "call_id": "call_orphan",
                "output": "result",
            },
            {"role": "user", "content": "Hello"},
        ]
        result = _remove_orphan_tool_responses(messages, {"call_orphan"})
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_function_call_output_non_orphan_kept(self):
        """Non-orphan function_call_output items should be kept."""
        messages = [
            {
                "type": "function_call_output",
                "call_id": "call_valid",
                "output": "result",
            },
            {"role": "user", "content": "Hello"},
        ]
        result = _remove_orphan_tool_responses(messages, {"call_other"})
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════
# validate_and_remove_orphan_tool_responses
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateOrphansResponsesApi:
    def test_chat_completions_paired_kept(self):
        """Baseline: matched Chat Completions pairs are kept."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ]
        result = validate_and_remove_orphan_tool_responses(messages, log_warning=False)
        assert len(result) == 2

    def test_responses_api_paired_kept(self):
        """Matched Responses API pairs are kept because the validator
        properly recognizes function_call and function_call_output items."""
        messages = [
            {"role": "user", "content": "Do something."},
            FUNCTION_CALL_ITEM,
            FUNCTION_CALL_OUTPUT_ITEM,
        ]
        result = validate_and_remove_orphan_tool_responses(messages, log_warning=False)
        assert len(result) == 3

    def test_responses_api_orphan_output_removed(self):
        """Orphan function_call_output (no matching function_call) should be removed."""
        messages = [
            {"role": "user", "content": "Do something."},
            # No function_call — output is orphaned
            FUNCTION_CALL_OUTPUT_ITEM,
        ]
        result = validate_and_remove_orphan_tool_responses(messages, log_warning=False)
        assert len(result) == 1
        assert result[0]["role"] == "user"


# ═══════════════════════════════════════════════════════════════════════════
# _ensure_tool_pairs_intact
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsureToolPairsIntactResponsesApi:
    def test_chat_completions_pair_preserved(self):
        """Baseline: sliced Chat Completions tool responses get their assistant prepended."""
        all_msgs = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "thanks"},
        ]
        # Slice starts at index 2 (tool response) — orphan
        recent = [all_msgs[2], all_msgs[3]]
        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index=2)
        # Should prepend the assistant message
        assert len(result) == 3
        assert "tool_calls" in result[0]

    def test_responses_api_pair_preserved(self):
        """Sliced function_call_output should get its function_call prepended."""
        all_msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "search for X"},
            FUNCTION_CALL_ITEM,
            FUNCTION_CALL_OUTPUT_ITEM,
            {"role": "user", "content": "thanks"},
        ]
        # Slice starts at index 3 (function_call_output) — orphan
        recent = [all_msgs[3], all_msgs[4]]
        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index=3)
        # Should prepend the function_call item
        assert len(result) == 3
        assert result[0].get("type") == "function_call"


# ═══════════════════════════════════════════════════════════════════════════
# _summarize_messages_llm (minor)
# ═══════════════════════════════════════════════════════════════════════════


class TestSummarizeMessagesResponsesApi:
    """_summarize_messages_llm extracts content using msg.get("role") and
    msg.get("content").  Responses API function_call items have neither
    role in ("user", "assistant", "tool") nor "content" — they'd be silently
    skipped in the summary.  This is a minor data-loss issue."""

    @pytest.mark.asyncio
    async def test_function_call_included_in_summary_text(self):
        """function_call items should contribute to the summary text."""
        from backend.util.prompt import _summarize_messages_llm

        messages = [
            {"role": "user", "content": "Search for X"},
            FUNCTION_CALL_ITEM,
            FUNCTION_CALL_OUTPUT_ITEM,
            {"role": "user", "content": "Thanks"},
        ]

        # We only need to check the conversation text building, not the LLM call.
        # The function builds conversation_text before calling the client.
        # We mock the client to capture what it receives.
        from unittest.mock import AsyncMock, MagicMock

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Summary"
        mock_client.with_options.return_value.chat.completions.create = AsyncMock(
            return_value=mock_resp
        )

        await _summarize_messages_llm(messages, mock_client, "gpt-4o")

        # Check the prompt sent to the LLM contains tool info
        call_args = (
            mock_client.with_options.return_value.chat.completions.create.call_args
        )
        user_msg = call_args.kwargs["messages"][1]["content"]
        # The tool name or arguments should appear in the summary text
        assert "search_tool" in user_msg or "python asyncio" in user_msg


# ═══════════════════════════════════════════════════════════════════════════
# compress_context end-to-end
# ═══════════════════════════════════════════════════════════════════════════


class TestCompressContextResponsesApi:
    @pytest.mark.asyncio
    async def test_chat_completions_tool_pairs_preserved(self):
        """Baseline: Chat Completions tool pairs survive compaction."""
        messages: list[dict] = [
            {"role": "system", "content": "You are helpful."},
        ]
        # Add enough messages to trigger compaction
        for i in range(20):
            messages.append({"role": "user", "content": f"Question {i} " * 200})
            messages.append({"role": "assistant", "content": f"Answer {i} " * 200})
        # Add a tool pair at the end
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_final", "type": "function", "function": {"name": "f"}}
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": "call_final", "content": "result"}
        )
        messages.append({"role": "assistant", "content": "Done!"})

        result = await compress_context(messages, target_tokens=2000, client=None)

        # If tool response exists, its call must exist too
        call_ids = set()
        resp_ids = set()
        for msg in result.messages:
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    call_ids.add(tc["id"])
            if msg.get("role") == "tool":
                resp_ids.add(msg.get("tool_call_id"))
        assert resp_ids <= call_ids

    @pytest.mark.asyncio
    async def test_responses_api_tool_pairs_preserved(self):
        """Responses API function_call / function_call_output pairs must
        survive compaction intact.  Currently they can be silently deleted
        because _is_tool_message doesn't recognise them."""
        messages = [
            {"role": "system", "content": "You are helpful."},
        ]
        # Add enough messages to trigger compaction
        for i in range(20):
            messages.append({"role": "user", "content": f"Question {i} " * 200})
            messages.append({"role": "assistant", "content": f"Answer {i} " * 200})
        # Add a Responses API tool pair at the end
        messages.append(
            {
                "type": "function_call",
                "id": "fc_final",
                "call_id": "call_final",
                "name": "search_tool",
                "arguments": '{"q": "test"}',
                "status": "completed",
            }
        )
        messages.append(
            {
                "type": "function_call_output",
                "call_id": "call_final",
                "output": '{"results": ["a", "b"]}',
            }
        )
        messages.append({"role": "user", "content": "Thanks!"})

        result = await compress_context(messages, target_tokens=2000, client=None)

        # The function_call and function_call_output must both survive
        fc_items = [m for m in result.messages if m.get("type") == "function_call"]
        fco_items = [
            m for m in result.messages if m.get("type") == "function_call_output"
        ]

        # If either exists, the other must exist too (pair integrity)
        if fc_items or fco_items:
            fc_call_ids = {m["call_id"] for m in fc_items}
            fco_call_ids = {m["call_id"] for m in fco_items}
            assert (
                fco_call_ids <= fc_call_ids
            ), "function_call_output exists without matching function_call"

        # At minimum, neither should have been silently deleted if the
        # conversation was short enough to keep them
        assert len(fc_items) >= 1, "function_call was deleted during compaction"
        assert len(fco_items) >= 1, "function_call_output was deleted during compaction"
