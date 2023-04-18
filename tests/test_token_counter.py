import unittest

import tests.context
from autogpt.token_counter import count_message_tokens, count_string_tokens


class TestTokenCounter(unittest.TestCase):
    def test_count_message_tokens(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        self.assertEqual(count_message_tokens(messages), 17)

    def test_count_message_tokens_with_name(self):
        messages = [
            {"role": "user", "content": "Hello", "name": "John"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        self.assertEqual(count_message_tokens(messages), 17)

    def test_count_message_tokens_empty_input(self):
        self.assertEqual(count_message_tokens([]), 3)

    def test_count_message_tokens_invalid_model(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        with self.assertRaises(KeyError):
            count_message_tokens(messages, model="invalid_model")

    def test_count_message_tokens_gpt_4(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        self.assertEqual(count_message_tokens(messages, model="gpt-4-0314"), 15)

    def test_count_string_tokens(self):
        string = "Hello, world!"
        self.assertEqual(
            count_string_tokens(string, model_name="gpt-3.5-turbo-0301"), 4
        )

    def test_count_string_tokens_empty_input(self):
        self.assertEqual(count_string_tokens("", model_name="gpt-3.5-turbo-0301"), 0)

    def test_count_message_tokens_invalid_model(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        with self.assertRaises(NotImplementedError):
            count_message_tokens(messages, model="invalid_model")

    def test_count_string_tokens_gpt_4(self):
        string = "Hello, world!"
        self.assertEqual(count_string_tokens(string, model_name="gpt-4-0314"), 4)


if __name__ == "__main__":
    unittest.main()
