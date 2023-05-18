import pytest

from autogpt.llm import count_message_tokens, count_string_tokens


def test_count_message_tokens():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert count_message_tokens(messages) == 17


def test_count_message_tokens_with_name():
    messages = [
        {"role": "user", "content": "Hello", "name": "John"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert count_message_tokens(messages) == 17


def test_count_message_tokens_empty_input():
    """Empty input should return 3 tokens"""
    assert count_message_tokens([]) == 3


def test_count_message_tokens_invalid_model():
    """Invalid model should raise a KeyError"""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    with pytest.raises(KeyError):
        count_message_tokens(messages, model="invalid_model")


def test_count_message_tokens_gpt_4():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert count_message_tokens(messages, model="gpt-4-0314") == 15


def test_count_string_tokens():
    """Test that the string tokens are counted correctly."""

    string = "Hello, world!"
    assert count_string_tokens(string, model_name="gpt-3.5-turbo-0301") == 4


def test_count_string_tokens_empty_input():
    """Test that the string tokens are counted correctly."""

    assert count_string_tokens("", model_name="gpt-3.5-turbo-0301") == 0


def test_count_message_tokens_invalid_model():
    """Invalid model should raise a NotImplementedError"""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    with pytest.raises(NotImplementedError):
        count_message_tokens(messages, model="invalid_model")


def test_count_string_tokens_gpt_4():
    """Test that the string tokens are counted correctly."""

    string = "Hello, world!"
    assert count_string_tokens(string, model_name="gpt-4-0314") == 4
