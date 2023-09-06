import pytest

from autogpt.llm.base import Message
from autogpt.llm.utils import count_message_tokens, count_string_tokens


def test_count_message_tokens():
    messages = [
        Message("user", "Hello"),
        Message("assistant", "Hi there!"),
    ]
    assert count_message_tokens(messages) == 17


def test_count_message_tokens_empty_input():
    """Empty input should return 3 tokens"""
    assert count_message_tokens([]) == 3


def test_count_message_tokens_invalid_model():
    """Invalid model should raise a NotImplementedError"""
    messages = [
        Message("user", "Hello"),
        Message("assistant", "Hi there!"),
    ]
    with pytest.raises(NotImplementedError):
        count_message_tokens(messages, model="invalid_model")


def test_count_message_tokens_gpt_4():
    messages = [
        Message("user", "Hello"),
        Message("assistant", "Hi there!"),
    ]
    assert count_message_tokens(messages, model="gpt-4-0314") == 15


def test_count_string_tokens():
    """Test that the string tokens are counted correctly."""

    string = "Hello, world!"
    assert count_string_tokens(string, model_name="gpt-3.5-turbo-0301") == 4


def test_count_string_tokens_empty_input():
    """Test that the string tokens are counted correctly."""

    assert count_string_tokens("", model_name="gpt-3.5-turbo-0301") == 0


def test_count_string_tokens_gpt_4():
    """Test that the string tokens are counted correctly."""

    string = "Hello, world!"
    assert count_string_tokens(string, model_name="gpt-4-0314") == 4
