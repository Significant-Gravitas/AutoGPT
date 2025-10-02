"""Tests for prompt utility functions, especially tool call token counting."""

import pytest
from tiktoken import encoding_for_model

from backend.util import json
from backend.util.prompt import _msg_tokens, estimate_token_count


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
