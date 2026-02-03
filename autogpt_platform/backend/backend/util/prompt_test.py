"""Tests for prompt utility functions, especially tool call token counting."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from tiktoken import encoding_for_model

from backend.util import json
from backend.util.prompt import (
    CompressResult,
    _ensure_tool_pairs_intact,
    _msg_tokens,
    _normalize_model_for_tokenizer,
    _truncate_middle_tokens,
    _truncate_tool_message_content,
    compress_context,
    estimate_token_count,
)


class TestMsgTokens:
    """Test the _msg_tokens function with various message types."""

    @pytest.fixture
    def enc(self):
        """Get the encoding for gpt-4o model."""
        return encoding_for_model("gpt-4o")

    def test_regular_message_token_counting(self, enc):
        """Test that regular messages are counted correctly (backward compatibility)."""
        msg = {"role": "user", "content": "What's the weather like in San Francisco?"}

        tokens = _msg_tokens(msg, enc)

        # Should be wrapper (3) + content tokens
        expected = 3 + len(enc.encode(msg["content"]))
        assert tokens == expected
        assert tokens > 3  # Has content

    def test_regular_message_with_name(self, enc):
        """Test that messages with name field get extra wrapper token."""
        msg = {"role": "user", "name": "test_user", "content": "Hello!"}

        tokens = _msg_tokens(msg, enc)

        # Should be wrapper (3 + 1 for name) + content tokens
        expected = 4 + len(enc.encode(msg["content"]))
        assert tokens == expected

    def test_openai_tool_call_token_counting(self, enc):
        """Test OpenAI format tool call token counting."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                    },
                }
            ],
        }

        tokens = _msg_tokens(msg, enc)

        # Should count wrapper + all tool call components
        expected_tool_tokens = (
            len(enc.encode("call_abc123"))
            + len(enc.encode("function"))
            + len(enc.encode("get_weather"))
            + len(enc.encode('{"location": "San Francisco", "unit": "celsius"}'))
        )
        expected = 3 + expected_tool_tokens  # wrapper + tool tokens

        assert tokens == expected
        assert tokens > 8  # Should be significantly more than just wrapper

    def test_openai_multiple_tool_calls(self, enc):
        """Test OpenAI format with multiple tool calls."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "func1", "arguments": '{"arg": "value1"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "func2", "arguments": '{"arg": "value2"}'},
                },
            ],
        }

        tokens = _msg_tokens(msg, enc)

        # Should count all tool calls
        assert tokens > 20  # Should be more than single tool call

    def test_anthropic_tool_use_token_counting(self, enc):
        """Test Anthropic format tool use token counting."""
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_xyz456",
                    "name": "get_weather",
                    "input": {"location": "San Francisco", "unit": "celsius"},
                }
            ],
        }

        tokens = _msg_tokens(msg, enc)

        # Should count wrapper + tool use components
        expected_tool_tokens = (
            len(enc.encode("toolu_xyz456"))
            + len(enc.encode("get_weather"))
            + len(
                enc.encode(json.dumps({"location": "San Francisco", "unit": "celsius"}))
            )
        )
        expected = 3 + expected_tool_tokens  # wrapper + tool tokens

        assert tokens == expected
        assert tokens > 8  # Should be significantly more than just wrapper

    def test_anthropic_tool_result_token_counting(self, enc):
        """Test Anthropic format tool result token counting."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_xyz456",
                    "content": "The weather in San Francisco is 22°C and sunny.",
                }
            ],
        }

        tokens = _msg_tokens(msg, enc)

        # Should count wrapper + tool result components
        expected_tool_tokens = len(enc.encode("toolu_xyz456")) + len(
            enc.encode("The weather in San Francisco is 22°C and sunny.")
        )
        expected = 3 + expected_tool_tokens  # wrapper + tool tokens

        assert tokens == expected
        assert tokens > 8  # Should be significantly more than just wrapper

    def test_anthropic_mixed_content(self, enc):
        """Test Anthropic format with mixed content types."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "content": "I'll check the weather for you."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "SF"},
                },
            ],
        }

        tokens = _msg_tokens(msg, enc)

        # Should count all content items
        assert tokens > 15  # Should count both text and tool use

    def test_empty_content(self, enc):
        """Test message with empty or None content."""
        msg = {"role": "assistant", "content": None}

        tokens = _msg_tokens(msg, enc)
        assert tokens == 3  # Just wrapper tokens

        msg["content"] = ""
        tokens = _msg_tokens(msg, enc)
        assert tokens == 3  # Just wrapper tokens

    def test_string_content_with_tool_calls(self, enc):
        """Test OpenAI format where content is string but tool_calls exist."""
        msg = {
            "role": "assistant",
            "content": "Let me check that for you.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_func", "arguments": "{}"},
                }
            ],
        }

        tokens = _msg_tokens(msg, enc)

        # Should count both content and tool calls
        content_tokens = len(enc.encode("Let me check that for you."))
        tool_tokens = (
            len(enc.encode("call_123"))
            + len(enc.encode("function"))
            + len(enc.encode("test_func"))
            + len(enc.encode("{}"))
        )
        expected = 3 + content_tokens + tool_tokens

        assert tokens == expected


class TestEstimateTokenCount:
    """Test the estimate_token_count function with conversations containing tool calls."""

    def test_conversation_with_tool_calls(self):
        """Test token counting for a complete conversation with tool calls."""
        conversation = [
            {"role": "user", "content": "What's the weather like in San Francisco?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "22°C and sunny",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The weather in San Francisco is 22°C and sunny.",
            },
        ]

        total_tokens = estimate_token_count(conversation)

        # Verify total equals sum of individual messages
        enc = encoding_for_model("gpt-4o")
        expected_total = sum(_msg_tokens(msg, enc) for msg in conversation)

        assert total_tokens == expected_total
        assert total_tokens > 40  # Should be substantial for this conversation

    def test_openai_conversation(self):
        """Test token counting for OpenAI format conversation."""
        conversation = [
            {"role": "user", "content": "Calculate 2 + 2"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_calc",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expression": "2 + 2"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_calc", "content": "4"},
            {"role": "assistant", "content": "The result is 4."},
        ]

        total_tokens = estimate_token_count(conversation)

        # Verify total equals sum of individual messages
        enc = encoding_for_model("gpt-4o")
        expected_total = sum(_msg_tokens(msg, enc) for msg in conversation)

        assert total_tokens == expected_total
        assert total_tokens > 20  # Should be substantial


class TestNormalizeModelForTokenizer:
    """Test model name normalization for tiktoken."""

    def test_openai_models_unchanged(self):
        """Test that OpenAI models are returned as-is."""
        assert _normalize_model_for_tokenizer("gpt-4o") == "gpt-4o"
        assert _normalize_model_for_tokenizer("gpt-4") == "gpt-4"
        assert _normalize_model_for_tokenizer("gpt-3.5-turbo") == "gpt-3.5-turbo"

    def test_claude_models_normalized(self):
        """Test that Claude models are normalized to gpt-4o."""
        assert _normalize_model_for_tokenizer("claude-3-opus") == "gpt-4o"
        assert _normalize_model_for_tokenizer("claude-3-sonnet") == "gpt-4o"
        assert _normalize_model_for_tokenizer("anthropic/claude-3-haiku") == "gpt-4o"

    def test_openrouter_paths_extracted(self):
        """Test that OpenRouter model paths are handled."""
        assert _normalize_model_for_tokenizer("openai/gpt-4o") == "gpt-4o"
        assert _normalize_model_for_tokenizer("anthropic/claude-3-opus") == "gpt-4o"

    def test_unknown_models_default_to_gpt4o(self):
        """Test that unknown models default to gpt-4o."""
        assert _normalize_model_for_tokenizer("some-random-model") == "gpt-4o"
        assert _normalize_model_for_tokenizer("llama-3-70b") == "gpt-4o"


class TestTruncateToolMessageContent:
    """Test tool message content truncation."""

    @pytest.fixture
    def enc(self):
        return encoding_for_model("gpt-4o")

    def test_truncate_openai_tool_message(self, enc):
        """Test truncation of OpenAI-style tool message with string content."""
        long_content = "x" * 10000
        msg = {"role": "tool", "tool_call_id": "call_123", "content": long_content}

        _truncate_tool_message_content(msg, enc, max_tokens=100)

        # Content should be truncated
        assert len(msg["content"]) < len(long_content)
        assert "…" in msg["content"]  # Has ellipsis marker

    def test_truncate_anthropic_tool_result(self, enc):
        """Test truncation of Anthropic-style tool_result."""
        long_content = "y" * 10000
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": long_content,
                }
            ],
        }

        _truncate_tool_message_content(msg, enc, max_tokens=100)

        # Content should be truncated
        result_content = msg["content"][0]["content"]
        assert len(result_content) < len(long_content)
        assert "…" in result_content

    def test_preserve_tool_use_blocks(self, enc):
        """Test that tool_use blocks are not truncated."""
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "some_function",
                    "input": {"key": "value" * 1000},  # Large input
                }
            ],
        }

        original = json.dumps(msg["content"][0]["input"])
        _truncate_tool_message_content(msg, enc, max_tokens=10)

        # tool_use should be unchanged
        assert json.dumps(msg["content"][0]["input"]) == original

    def test_no_truncation_when_under_limit(self, enc):
        """Test that short content is not modified."""
        msg = {"role": "tool", "tool_call_id": "call_123", "content": "Short content"}

        original = msg["content"]
        _truncate_tool_message_content(msg, enc, max_tokens=1000)

        assert msg["content"] == original


class TestTruncateMiddleTokens:
    """Test middle truncation of text."""

    @pytest.fixture
    def enc(self):
        return encoding_for_model("gpt-4o")

    def test_truncates_long_text(self, enc):
        """Test that long text is truncated with ellipsis in middle."""
        long_text = "word " * 1000
        result = _truncate_middle_tokens(long_text, enc, max_tok=50)

        assert len(enc.encode(result)) <= 52  # Allow some slack for ellipsis
        assert "…" in result
        assert result.startswith("word")  # Head preserved
        assert result.endswith("word ")  # Tail preserved

    def test_preserves_short_text(self, enc):
        """Test that short text is not modified."""
        short_text = "Hello world"
        result = _truncate_middle_tokens(short_text, enc, max_tok=100)

        assert result == short_text


class TestEnsureToolPairsIntact:
    """Test tool call/response pair preservation for both OpenAI and Anthropic formats."""

    # ---- OpenAI Format Tests ----

    def test_openai_adds_missing_tool_call(self):
        """Test that orphaned OpenAI tool_response gets its tool_call prepended."""
        all_msgs = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "f1"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "Thanks!"},
        ]
        # Recent messages start at index 2 (the tool response)
        recent = [all_msgs[2], all_msgs[3]]
        start_index = 2

        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index)

        # Should prepend the tool_call message
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]

    def test_openai_keeps_complete_pairs(self):
        """Test that complete OpenAI pairs are unchanged."""
        all_msgs = [
            {"role": "system", "content": "System"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "f1"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        recent = all_msgs[1:]  # Include both tool_call and response
        start_index = 1

        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index)

        assert len(result) == 2  # No messages added

    def test_openai_multiple_tool_calls(self):
        """Test multiple OpenAI tool calls in one assistant message."""
        all_msgs = [
            {"role": "system", "content": "System"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "f1"}},
                    {"id": "call_2", "type": "function", "function": {"name": "f2"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result2"},
            {"role": "user", "content": "Thanks!"},
        ]
        # Recent messages start at index 2 (first tool response)
        recent = [all_msgs[2], all_msgs[3], all_msgs[4]]
        start_index = 2

        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index)

        # Should prepend the assistant message with both tool_calls
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 2

    # ---- Anthropic Format Tests ----

    def test_anthropic_adds_missing_tool_use(self):
        """Test that orphaned Anthropic tool_result gets its tool_use prepended."""
        all_msgs = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"location": "SF"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "22°C and sunny",
                    }
                ],
            },
            {"role": "user", "content": "Thanks!"},
        ]
        # Recent messages start at index 2 (the tool_result)
        recent = [all_msgs[2], all_msgs[3]]
        start_index = 2

        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index)

        # Should prepend the tool_use message
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "tool_use"

    def test_anthropic_keeps_complete_pairs(self):
        """Test that complete Anthropic pairs are unchanged."""
        all_msgs = [
            {"role": "system", "content": "System"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_456",
                        "name": "calculator",
                        "input": {"expr": "2+2"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_456",
                        "content": "4",
                    }
                ],
            },
        ]
        recent = all_msgs[1:]  # Include both tool_use and result
        start_index = 1

        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index)

        assert len(result) == 2  # No messages added

    def test_anthropic_multiple_tool_uses(self):
        """Test multiple Anthropic tool_use blocks in one message."""
        all_msgs = [
            {"role": "system", "content": "System"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check both..."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "get_weather",
                        "input": {"city": "LA"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "Cold",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": "Warm",
                    },
                ],
            },
            {"role": "user", "content": "Thanks!"},
        ]
        # Recent messages start at index 2 (tool_result)
        recent = [all_msgs[2], all_msgs[3]]
        start_index = 2

        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index)

        # Should prepend the assistant message with both tool_uses
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        tool_use_count = sum(
            1 for b in result[0]["content"] if b.get("type") == "tool_use"
        )
        assert tool_use_count == 2

    # ---- Mixed/Edge Case Tests ----

    def test_anthropic_with_type_message_field(self):
        """Test Anthropic format with 'type': 'message' field (smart_decision_maker style)."""
        all_msgs = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "search",
                        "input": {"q": "test"},
                    }
                ],
            },
            {
                "role": "user",
                "type": "message",  # Extra field from smart_decision_maker
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "Found results",
                    }
                ],
            },
            {"role": "user", "content": "Thanks!"},
        ]
        # Recent messages start at index 2 (the tool_result with 'type': 'message')
        recent = [all_msgs[2], all_msgs[3]]
        start_index = 2

        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index)

        # Should prepend the tool_use message
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "tool_use"

    def test_handles_no_tool_messages(self):
        """Test messages without tool calls."""
        all_msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        recent = all_msgs
        start_index = 0

        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index)

        assert result == all_msgs

    def test_handles_empty_messages(self):
        """Test empty message list."""
        result = _ensure_tool_pairs_intact([], [], 0)
        assert result == []

    def test_mixed_text_and_tool_content(self):
        """Test Anthropic message with mixed text and tool_use content."""
        all_msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll help you with that."},
                    {
                        "type": "tool_use",
                        "id": "toolu_mixed",
                        "name": "search",
                        "input": {"q": "test"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_mixed",
                        "content": "Found results",
                    }
                ],
            },
            {"role": "assistant", "content": "Here are the results..."},
        ]
        # Start from tool_result
        recent = [all_msgs[1], all_msgs[2]]
        start_index = 1

        result = _ensure_tool_pairs_intact(recent, all_msgs, start_index)

        # Should prepend the assistant message with tool_use
        assert len(result) == 3
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "tool_use"


class TestCompressContext:
    """Test the async compress_context function."""

    @pytest.mark.asyncio
    async def test_no_compression_needed(self):
        """Test messages under limit return without compression."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        result = await compress_context(messages, target_tokens=100000)

        assert isinstance(result, CompressResult)
        assert result.was_compacted is False
        assert len(result.messages) == 2
        assert result.error is None

    @pytest.mark.asyncio
    async def test_truncation_without_client(self):
        """Test that truncation works without LLM client."""
        long_content = "x" * 50000
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "Response"},
        ]

        result = await compress_context(
            messages, target_tokens=1000, client=None, reserve=100
        )

        assert result.was_compacted is True
        # Should have truncated without summarization
        assert result.messages_summarized == 0

    @pytest.mark.asyncio
    async def test_with_mocked_llm_client(self):
        """Test summarization with mocked LLM client."""
        # Create many messages to trigger summarization
        messages = [{"role": "system", "content": "System prompt"}]
        for i in range(30):
            messages.append({"role": "user", "content": f"User message {i} " * 100})
            messages.append(
                {"role": "assistant", "content": f"Assistant response {i} " * 100}
            )

        # Mock the AsyncOpenAI client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of conversation"
        mock_client.with_options.return_value.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await compress_context(
            messages,
            target_tokens=5000,
            client=mock_client,
            keep_recent=5,
            reserve=500,
        )

        assert result.was_compacted is True
        # Should have attempted summarization
        assert mock_client.with_options.called or result.messages_summarized > 0

    @pytest.mark.asyncio
    async def test_preserves_tool_pairs(self):
        """Test that tool call/response pairs stay together."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "func"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Result " * 1000},
            {"role": "assistant", "content": "Done!"},
        ]

        result = await compress_context(
            messages, target_tokens=500, client=None, reserve=50
        )

        # Check that if tool response exists, its call exists too
        tool_call_ids = set()
        tool_response_ids = set()
        for msg in result.messages:
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    tool_call_ids.add(tc["id"])
            if msg.get("role") == "tool":
                tool_response_ids.add(msg.get("tool_call_id"))

        # All tool responses should have their calls
        assert tool_response_ids <= tool_call_ids

    @pytest.mark.asyncio
    async def test_returns_error_when_cannot_compress(self):
        """Test that error is returned when compression fails."""
        # Single huge message that can't be compressed enough
        messages = [
            {"role": "user", "content": "x" * 100000},
        ]

        result = await compress_context(
            messages, target_tokens=100, client=None, reserve=50
        )

        # Should have an error since we can't get below 100 tokens
        assert result.error is not None
        assert result.was_compacted is True

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test that empty messages list returns early without error."""
        result = await compress_context([], target_tokens=1000)

        assert result.messages == []
        assert result.token_count == 0
        assert result.was_compacted is False
        assert result.error is None


class TestRemoveOrphanToolResponses:
    """Test _remove_orphan_tool_responses helper function."""

    def test_removes_openai_orphan(self):
        """Test removal of orphan OpenAI tool response."""
        from backend.util.prompt import _remove_orphan_tool_responses

        messages = [
            {"role": "tool", "tool_call_id": "call_orphan", "content": "result"},
            {"role": "user", "content": "Hello"},
        ]
        orphan_ids = {"call_orphan"}

        result = _remove_orphan_tool_responses(messages, orphan_ids)

        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_keeps_valid_openai_tool(self):
        """Test that valid OpenAI tool responses are kept."""
        from backend.util.prompt import _remove_orphan_tool_responses

        messages = [
            {"role": "tool", "tool_call_id": "call_valid", "content": "result"},
        ]
        orphan_ids = {"call_other"}

        result = _remove_orphan_tool_responses(messages, orphan_ids)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_valid"

    def test_filters_anthropic_mixed_blocks(self):
        """Test filtering individual orphan blocks from Anthropic message with mixed valid/orphan."""
        from backend.util.prompt import _remove_orphan_tool_responses

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_valid",
                        "content": "valid result",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_orphan",
                        "content": "orphan result",
                    },
                ],
            },
        ]
        orphan_ids = {"toolu_orphan"}

        result = _remove_orphan_tool_responses(messages, orphan_ids)

        assert len(result) == 1
        # Should only have the valid tool_result, orphan filtered out
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["tool_use_id"] == "toolu_valid"

    def test_removes_anthropic_all_orphan(self):
        """Test removal of Anthropic message when all tool_results are orphans."""
        from backend.util.prompt import _remove_orphan_tool_responses

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_orphan1",
                        "content": "result1",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_orphan2",
                        "content": "result2",
                    },
                ],
            },
        ]
        orphan_ids = {"toolu_orphan1", "toolu_orphan2"}

        result = _remove_orphan_tool_responses(messages, orphan_ids)

        # Message should be completely removed since no content left
        assert len(result) == 0

    def test_preserves_non_tool_messages(self):
        """Test that non-tool messages are preserved."""
        from backend.util.prompt import _remove_orphan_tool_responses

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        orphan_ids = {"some_id"}

        result = _remove_orphan_tool_responses(messages, orphan_ids)

        assert result == messages


class TestCompressResultDataclass:
    """Test CompressResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = CompressResult(
            messages=[{"role": "user", "content": "test"}],
            token_count=10,
            was_compacted=False,
        )

        assert result.error is None
        assert result.original_token_count == 0  # Defaults to 0, not None
        assert result.messages_summarized == 0
        assert result.messages_dropped == 0

    def test_all_fields(self):
        """Test all fields can be set."""
        result = CompressResult(
            messages=[{"role": "user", "content": "test"}],
            token_count=100,
            was_compacted=True,
            error="Some error",
            original_token_count=500,
            messages_summarized=10,
            messages_dropped=5,
        )

        assert result.token_count == 100
        assert result.was_compacted is True
        assert result.error == "Some error"
        assert result.original_token_count == 500
        assert result.messages_summarized == 10
        assert result.messages_dropped == 5
