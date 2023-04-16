from unittest import TestCase

from autogpt.json_fixes.parsing import fix_and_parse_json


class TestParseJson(TestCase):
    def test_valid_json(self):
        # Test that a valid JSON string is parsed correctly
        json_str = '{"name": "John", "age": 30, "city": "New York"}'
        obj = fix_and_parse_json(json_str)
        self.assertEqual(obj, {"name": "John", "age": 30, "city": "New York"})

    def test_invalid_json_minor(self):
        # Test that an invalid JSON string can be fixed with gpt
        json_str = '{"name": "John", "age": 30, "city": "New York",}'
        with self.assertRaises(Exception):
            fix_and_parse_json(json_str, try_to_fix_with_gpt=False)

    def test_invalid_json_major_with_gpt(self):
        pass

    def test_invalid_json_major_without_gpt(self):
        # Test that a REALLY invalid JSON string raises an error when try_to_fix_with_gpt is False
        json_str = 'BEGIN: "name": "John" - "age": 30 - "city": "New York" :END'
        # Assert that this raises an exception:
        with self.assertRaises(Exception):
            fix_and_parse_json(json_str, try_to_fix_with_gpt=False)

    def test_invalid_json_leading_sentence(self):
        json_str = """I will first need to browse the repository (https://github.com/Torantulino/Auto-GPT) and identify any potential bugs that need fixing. I will use the "browse_website" command for this.

{
    "command": {
        "name": "browse_website",
        "args":{
            "url": "https://github.com/Torantulino/Auto-GPT"
        }
    },
    "thoughts":
    {
        "text": "Browsing the repository to identify potential bugs",
        "reasoning": "Before fixing bugs, I need to identify what needs fixing. I will use the 'browse_website' command to analyze the repository.",
        "plan": "- Analyze the repository for potential bugs and areas of improvement",
        "criticism": "I need to ensure I am thorough and pay attention to detail while browsing the repository.",
        "speak": "I am browsing the repository to identify potential bugs."
    }
}"""
        good_obj = {
            "command": {
                "name": "browse_website",
                "args": {"url": "https://github.com/Torantulino/Auto-GPT"},
            },
            "thoughts": {
                "text": "Browsing the repository to identify potential bugs",
                "reasoning": "Before fixing bugs, I need to identify what needs fixing. I will use the 'browse_website' command to analyze the repository.",
                "plan": "- Analyze the repository for potential bugs and areas of improvement",
                "criticism": "I need to ensure I am thorough and pay attention to detail while browsing the repository.",
                "speak": "I am browsing the repository to identify potential bugs.",
            },
        }
        self.assertEqual(fix_and_parse_json(json_str), good_obj)
