from unittest import TestCase

from autogpt.json_utils.json_fix_llm import fix_and_parse_json
from tests.utils import skip_in_ci


class TestParseJson(TestCase):
    def test_valid_json(self):
        """Test that a valid JSON string is parsed correctly."""
        json_str = '{"name": "John", "age": 30, "city": "New York"}'
        obj = fix_and_parse_json(json_str)
        self.assertEqual(obj, {"name": "John", "age": 30, "city": "New York"})

    def test_invalid_json_minor(self):
        """Test that an invalid JSON string can not be fixed without gpt"""
        json_str = '{"name": "John", "age": 30, "city": "New York",}'
        with self.assertRaises(Exception):
            fix_and_parse_json(json_str, try_to_fix_with_gpt=False)

    def test_invalid_json_major_with_gpt(self):
        """Test that an invalid JSON string raises an error when try_to_fix_with_gpt is False"""
        json_str = 'BEGIN: "name": "John" - "age": 30 - "city": "New York" :END'
        with self.assertRaises(Exception):
            fix_and_parse_json(json_str, try_to_fix_with_gpt=False)

    def test_invalid_json_major_without_gpt(self):
        """Test that a REALLY invalid JSON string raises an error when try_to_fix_with_gpt is False"""
        json_str = 'BEGIN: "name": "John" - "age": 30 - "city": "New York" :END'
        # Assert that this raises an exception:
        with self.assertRaises(Exception):
            fix_and_parse_json(json_str, try_to_fix_with_gpt=False)

    def test_invalid_json_leading_sentence_with_gpt(self):
        """Test that a REALLY invalid JSON string raises an error when try_to_fix_with_gpt is False"""
        json_str = """I suggest we start by browsing the repository to find any issues that we can fix.

{
    "command": {
        "name": "browse_website",
        "args":{
            "url": "https://github.com/Torantulino/Auto-GPT"
        }
    },
    "thoughts":
    {
        "text": "I suggest we start browsing the repository to find any issues that we can fix.",
        "reasoning": "Browsing the repository will give us an idea of the current state of the codebase and identify any issues that we can address to improve the repo.",
        "plan": "- Look through the repository to find any issues.\n- Investigate any issues to determine what needs to be fixed\n- Identify possible solutions to fix the issues\n- Open Pull Requests with fixes",
        "criticism": "I should be careful while browsing so as not to accidentally introduce any new bugs or issues.",
        "speak": "I will start browsing the repository to find any issues we can fix."
    }
}"""
        good_obj = {
            "command": {
                "name": "browse_website",
                "args": {"url": "https://github.com/Torantulino/Auto-GPT"},
            },
            "thoughts": {
                "text": "I suggest we start browsing the repository to find any issues that we can fix.",
                "reasoning": "Browsing the repository will give us an idea of the current state of the codebase and identify any issues that we can address to improve the repo.",
                "plan": "- Look through the repository to find any issues.\n- Investigate any issues to determine what needs to be fixed\n- Identify possible solutions to fix the issues\n- Open Pull Requests with fixes",
                "criticism": "I should be careful while browsing so as not to accidentally introduce any new bugs or issues.",
                "speak": "I will start browsing the repository to find any issues we can fix.",
            },
        }

        # # Assert that this can be fixed with GPT
        # self.assertEqual(fix_and_parse_json(json_str), good_obj)

        # Assert that trying to fix this without GPT raises an exception
        with self.assertRaises(Exception):
            fix_and_parse_json(json_str, try_to_fix_with_gpt=False)
