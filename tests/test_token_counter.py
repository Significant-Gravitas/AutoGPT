import unittest

import tests.context
from autogpt.token_counter import count_message_tokens, count_string_tokens


class TestTokenCounter(unittest.TestCase):
    """Test the token counter."""

    def test_count_message_tokens(self):
        """Test that the message tokens are counted correctly."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        self.assertEqual(count_message_tokens(messages), 17)

    def test_count_message_tokens_with_name(self):
        """Test that the name is counted as a token."""
        messages = [
            {"role": "user", "content": "Hello", "name": "John"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        self.assertEqual(count_message_tokens(messages), 17)

    def test_count_message_tokens_empty_input(self):
        """Test that the message tokens are counted correctly."""
        # Empty input should return 3 tokens
        self.assertEqual(count_message_tokens([]), 3)

    def test_count_message_tokens_invalid_model(self):
        """Test that the message tokens are counted correctly."""
        # Invalid model should raise a KeyError
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        with self.assertRaises(KeyError):
            count_message_tokens(messages, model="invalid_model")

    def test_count_message_tokens_gpt_4(self):
        """Test that the message tokens are counted correctly."""
        # GPT-4 should count 15 tokens
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        self.assertEqual(count_message_tokens(messages, model="gpt-4-0314"), 15)

    def test_count_string_tokens(self):
        """Test that the string tokens are counted correctly."""
        # GPT-3.5-turbo-0301 should count 4 tokens
        string = "Hello, world!"
        self.assertEqual(
            count_string_tokens(string, model_name="gpt-3.5-turbo-0301"), 4
        )

    def test_count_string_tokens_empty_input(self):
        """Test that the string tokens are counted correctly."""
        # GPT-3.5-turbo-0301 should count 0 tokens for empty input
        self.assertEqual(count_string_tokens("", model_name="gpt-3.5-turbo-0301"), 0)

    def test_count_message_tokens_invalid_model(self):
        """Test that the message tokens are counted correctly."""
        # Invalid model should raise a NotImplementedError
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        with self.assertRaises(NotImplementedError):
            count_message_tokens(messages, model="invalid_model")

    def test_count_string_tokens_gpt_4(self):
        """Test that the string tokens are counted correctly."""
        # GPT-4 should count 4 tokens
        string = "Hello, world!"
        self.assertEqual(count_string_tokens(string, model_name="gpt-4-0314"), 4)


if __name__ == "__main__":
    unittest.main()
