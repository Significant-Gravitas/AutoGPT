import unittest
from unittest.mock import patch
from typing import List
from src.refactoring_tool.ai_functions import call_ai_function, validate_and_sanitize_input


class TestAIFunctions(unittest.TestCase):

    def test_validate_and_sanitize_input(self):
        valid_function = "def test_function(a, b):"
        valid_args = ["arg1", "arg2"]
        valid_description = "Test function description."

        self.assertTrue(validate_and_sanitize_input(valid_function, valid_args, valid_description))

        # Test invalid inputs
        self.assertFalse(validate_and_sanitize_input(None, valid_args, valid_description))
        self.assertFalse(validate_and_sanitize_input(valid_function, None, valid_description))
        self.assertFalse(validate_and_sanitize_input(valid_function, valid_args, None))

    @patch('src.refactoring_tool.ai_functions.openai.ChatCompletion.create')
    def test_call_ai_function(self, mock_chat_completion_create):
        function = "def test_function(a, b):"
        args = ["arg1", "arg2"]
        description = "Test function description."

        # Create a nested object structure that mimics the API response
        response_obj = lambda: None
        response_obj.choices = [lambda: None]
        response_obj.choices[0].message = {"content": "Test AI response"}

        mock_chat_completion_create.return_value = response_obj

        response = call_ai_function(function, tuple(args), description)
        self.assertEqual(response, "Test AI response")

        # Test with invalid inputs
        with self.assertRaises(ValueError):
            call_ai_function(None, tuple(args), description)

        with self.assertRaises(ValueError):
            call_ai_function(function, None, description)

        with self.assertRaises(ValueError):
            call_ai_function(function, tuple(args), None)


if __name__ == '__main__':
    unittest.main()
