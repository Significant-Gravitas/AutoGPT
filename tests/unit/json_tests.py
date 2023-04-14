import unittest
import os
import sys


from autogpt.json_parser import fix_and_parse_json


class TestParseJson(unittest.TestCase):
    def test_valid_json(self):
        # Test that a valid JSON string is parsed correctly
        json_str = '{"name": "John", "age": 30, "city": "New York"}'
        obj = fix_and_parse_json(json_str)
        self.assertEqual(obj, {"name": "John", "age": 30, "city": "New York"})

    def test_invalid_json_minor(self):
        # Test that an invalid JSON string can be fixed with gpt
        json_str = '{"name": "John", "age": 30, "city": "New York",}'
        self.assertEqual(fix_and_parse_json(json_str, try_to_fix_with_gpt=False), {"name": "John", "age": 30, "city": "New York"})

    def test_invalid_json_major_with_gpt(self):
        # Test that an invalid JSON string raises an error when try_to_fix_with_gpt is False
        json_str = 'BEGIN: "name": "John" - "age": 30 - "city": "New York" :END'
        self.assertEqual(fix_and_parse_json(json_str, try_to_fix_with_gpt=True), {"name": "John", "age": 30, "city": "New York"})

    def test_invalid_json_major_without_gpt(self):
        # Test that a REALLY invalid JSON string raises an error when try_to_fix_with_gpt is False
        json_str = 'BEGIN: "name": "John" - "age": 30 - "city": "New York" :END'
        # Assert that this raises an exception:
        with self.assertRaises(Exception):
            fix_and_parse_json(json_str, try_to_fix_with_gpt=False)

    def test_invalid_json_leading_sentence_with_gpt(self):
        # Test that a REALLY invalid JSON string raises an error when try_to_fix_with_gpt is False
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
              "args": {
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
      }
        # Assert that this raises an exception:
        self.assertEqual(fix_and_parse_json(json_str, try_to_fix_with_gpt=False), good_obj)

    def test_invalid_json_leading_sentence_with_gpt(self):
        # Test that a REALLY invalid JSON string raises an error when try_to_fix_with_gpt is False
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
        "args": {
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
}
        # Assert that this raises an exception:
        self.assertEqual(fix_and_parse_json(json_str, try_to_fix_with_gpt=False), good_obj)

    def test_that_apologies_containing_multiple_json_get_the_correct_one(self):
        bad_json = 'I apologize once again for the error. Here is the corrected format to run the tests: ``` { "name": "execute_python_file", "args": { "file": "<test_file_location>" } } ``` Where `<test_file_location>` should be replaced with the file path to the test file you created in the previous step. For example: ``` { "name": "execute_python_file", "args": { "file": "tests/test_addition.py" } } ``` This will execute the tests for the `add_numbers` function in `tests/test_addition.py`. Please let me know if you have any further questions.'
        actual_json = fix_and_parse_json(bad_json, try_to_fix_with_gpt=True)
        expected_json = { "name": "execute_python_file", "args": { "file": "tests/test_addition.py" } }
        self.assertEqual(actual_json, expected_json)
        # TODO come back to fix this test after fixing imports


if __name__ == '__main__':
    unittest.main()
