"""Tests for get_pending_tool_calls function to prevent regression of ChatCompletionMessage handling."""

import pytest
from unittest.mock import MagicMock
from typing import Any

from backend.blocks.smart_decision_maker import (
    SmartDecisionMakerBlock,
    get_pending_tool_calls,
    _get_tool_requests,
    _get_tool_responses,
)


class MockChatCompletionMessage:
    """Mock object that simulates OpenAI's ChatCompletionMessage structure."""

    def __init__(self, role: str, content: str = None, tool_calls: list = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []

    def get(self, key: str, default: Any = None) -> Any:
        """This simulates what would cause the AttributeError in the original bug."""
        raise AttributeError(
            f"'MockChatCompletionMessage' object has no attribute 'get'"
        )


def test_get_pending_tool_calls_with_dict():
    """Test that get_pending_tool_calls works with regular dictionaries."""
    conversation_history = [
        {
            "role": "assistant",
            "content": "I'll help you with that.",
            "tool_calls": [{"id": "call_123", "function": {"name": "test_tool"}}],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"},
    ]

    pending_calls = get_pending_tool_calls(conversation_history)
    assert len(pending_calls) == 0  # All tool calls have responses


def test_get_pending_tool_calls_with_pending():
    """Test that get_pending_tool_calls correctly identifies pending tool calls."""
    conversation_history = [
        {
            "role": "assistant",
            "content": "I'll help you with that.",
            "tool_calls": [
                {"id": "call_123", "function": {"name": "test_tool"}},
                {"id": "call_456", "function": {"name": "another_tool"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"},
        # call_456 has no response, so it should be pending
    ]

    pending_calls = get_pending_tool_calls(conversation_history)
    assert len(pending_calls) == 1
    assert "call_456" in pending_calls
    assert pending_calls["call_456"] == 1


def test_get_pending_tool_calls_anthropic_format():
    """Test that get_pending_tool_calls works with Anthropic format."""
    conversation_history = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me help you with that."},
                {"type": "tool_use", "id": "toolu_123", "name": "test_tool"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Result"}
            ],
        },
    ]

    pending_calls = get_pending_tool_calls(conversation_history)
    assert len(pending_calls) == 0  # All tool calls have responses


def test_get_pending_tool_calls_anthropic_pending():
    """Test pending tool calls in Anthropic format."""
    conversation_history = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me help you with that."},
                {"type": "tool_use", "id": "toolu_123", "name": "test_tool"},
                {"type": "tool_use", "id": "toolu_456", "name": "another_tool"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Result"}
            ],
        },
    ]

    pending_calls = get_pending_tool_calls(conversation_history)
    assert len(pending_calls) == 1
    assert "toolu_456" in pending_calls


def test_get_pending_tool_calls_mixed_formats():
    """Test that mixed OpenAI and Anthropic formats work correctly."""
    conversation_history = [
        # OpenAI format tool request
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_123", "function": {"name": "openai_tool"}}],
        },
        # Anthropic format tool request
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_456", "name": "anthropic_tool"}],
        },
        # OpenAI format response
        {"role": "tool", "tool_call_id": "call_123", "content": "OpenAI result"},
        # Anthropic format response
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_456", "content": "Anthropic result"}
            ],
        },
    ]

    pending_calls = get_pending_tool_calls(conversation_history)
    assert len(pending_calls) == 0  # All tool calls have responses


def test_smart_decision_maker_get_missing_input_with_chat_completion_message():
    """
    Test that SmartDecisionMakerBlock.Input.get_missing_input handles ChatCompletionMessage objects.
    This is the regression test for the original bug.
    """
    # Create mock ChatCompletionMessage objects that would cause the original bug
    mock_message1 = MockChatCompletionMessage(
        role="assistant",
        content="I'll help you with that.",
        tool_calls=[{"id": "call_123", "function": {"name": "test_tool"}}],
    )
    mock_message2 = MockChatCompletionMessage(
        role="tool", content="Tool result"
    )

    # Create input data with mock ChatCompletionMessage objects
    input_data = {
        "conversation_history": [mock_message1, mock_message2],
        "last_tool_output": None,
    }

    # This should NOT raise AttributeError after the fix
    # The fix converts ChatCompletionMessage objects to dicts before calling get_pending_tool_calls
    missing_input = SmartDecisionMakerBlock.Input.get_missing_input(input_data)

    # Since the conversation history has ChatCompletionMessage objects that can't be processed
    # directly, the fix should handle them properly
    assert isinstance(missing_input, set)


def test_smart_decision_maker_get_missing_input_with_pending_tools():
    """Test get_missing_input correctly identifies when last_tool_output is needed."""
    input_data = {
        "conversation_history": [
            {
                "role": "assistant",
                "content": "I'll help you with that.",
                "tool_calls": [{"id": "call_123", "function": {"name": "test_tool"}}],
            }
            # No tool response, so tool call is pending
        ],
        "last_tool_output": None,
    }

    missing_input = SmartDecisionMakerBlock.Input.get_missing_input(input_data)
    assert "last_tool_output" in missing_input


def test_smart_decision_maker_get_missing_input_with_tool_output_no_pending():
    """Test get_missing_input when tool output exists but no pending calls."""
    input_data = {
        "conversation_history": [
            {
                "role": "assistant",
                "content": "Task complete.",
            }
        ],
        "last_tool_output": "Some output",  # Output exists but no pending calls
    }

    missing_input = SmartDecisionMakerBlock.Input.get_missing_input(input_data)
    assert "conversation_history" in missing_input


def test_smart_decision_maker_get_missing_input_empty_conversation():
    """Test get_missing_input with empty conversation history."""
    input_data = {
        "conversation_history": [],
        "last_tool_output": None,
    }

    missing_input = SmartDecisionMakerBlock.Input.get_missing_input(input_data)
    assert len(missing_input) == 0  # No missing input


def test_smart_decision_maker_get_missing_input_with_none_entries():
    """Test that None entries in conversation history are filtered out."""
    input_data = {
        "conversation_history": [
            None,
            {
                "role": "assistant",
                "content": "Hello",
            },
            None,
        ],
        "last_tool_output": None,
    }

    # Should not raise an error with None entries
    missing_input = SmartDecisionMakerBlock.Input.get_missing_input(input_data)
    assert len(missing_input) == 0


@pytest.mark.asyncio
async def test_smart_decision_maker_run_with_chat_completion_message():
    """
    Integration test that SmartDecisionMakerBlock.run handles ChatCompletionMessage objects.
    """
    from unittest.mock import patch, MagicMock
    import backend.blocks.llm as llm_module

    block = SmartDecisionMakerBlock()

    # Mock ChatCompletionMessage that would cause the original bug
    mock_message = MockChatCompletionMessage(
        role="user",
        content="Hello"
    )

    # Mock the response
    mock_response = MagicMock()
    mock_response.response = "Hi there!"
    mock_response.tool_calls = None
    mock_response.prompt_tokens = 10
    mock_response.completion_tokens = 5
    mock_response.reasoning = None
    mock_response.raw_response = {"role": "assistant", "content": "Hi there!"}

    with patch("backend.blocks.llm.llm_call", return_value=mock_response), \
         patch.object(SmartDecisionMakerBlock, "_create_function_signature", return_value=[]):

        input_data = SmartDecisionMakerBlock.Input(
            prompt="Test prompt",
            model=llm_module.LlmModel.GPT4O,
            credentials=llm_module.TEST_CREDENTIALS_INPUT,
            conversation_history=[mock_message],  # This would fail without the fix
        )

        # This should not raise AttributeError after the fix
        outputs = {}
        async for output_name, output_data in block.run(
            input_data,
            credentials=llm_module.TEST_CREDENTIALS,
            graph_id="test-graph-id",
            node_id="test-node-id",
            graph_exec_id="test-exec-id",
            node_exec_id="test-node-exec-id",
            user_id="test-user-id",
        ):
            outputs[output_name] = output_data

        assert "finished" in outputs
        assert outputs["finished"] == "Hi there!"


def test_get_tool_requests_openai_format():
    """Test _get_tool_requests with OpenAI format."""
    entry = {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_123", "function": {"name": "tool1"}},
            {"id": "call_456", "function": {"name": "tool2"}},
        ]
    }

    tool_ids = _get_tool_requests(entry)
    assert len(tool_ids) == 2
    assert "call_123" in tool_ids
    assert "call_456" in tool_ids


def test_get_tool_requests_anthropic_format():
    """Test _get_tool_requests with Anthropic format."""
    entry = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me help"},
            {"type": "tool_use", "id": "toolu_123", "name": "tool1"},
            {"type": "tool_use", "id": "toolu_456", "name": "tool2"},
        ]
    }

    tool_ids = _get_tool_requests(entry)
    assert len(tool_ids) == 2
    assert "toolu_123" in tool_ids
    assert "toolu_456" in tool_ids


def test_get_tool_responses_openai_format():
    """Test _get_tool_responses with OpenAI format."""
    entry = {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "Tool result"
    }

    tool_ids = _get_tool_responses(entry)
    assert len(tool_ids) == 1
    assert "call_123" in tool_ids


def test_get_tool_responses_anthropic_format():
    """Test _get_tool_responses with Anthropic format."""
    entry = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Result 1"},
            {"type": "tool_result", "tool_use_id": "toolu_456", "content": "Result 2"},
        ]
    }

    tool_ids = _get_tool_responses(entry)
    assert len(tool_ids) == 2
    assert "toolu_123" in tool_ids
    assert "toolu_456" in tool_ids