import unittest
from unittest.mock import MagicMock, patch
from typing import List, Optional
from src.refactoring_tool.ai_functions import AIFunctionCaller


class TestAIFunctionCaller(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a default instance of the AIFunctionCaller class for each test case.
        """
        self.ai_function_caller = AIFunctionCaller()

    def test_validate_and_sanitize_input(self):
        """
        Test the validate_and_sanitize_input method of the AIFunctionCaller class.
        """
        valid_function = "def test_function(a: str, b: str) -> str:"
        valid_args = ["arg1", "arg2"]
        valid_description = "Test description"

        self.assertTrue(self.ai_function_caller.validate_and_sanitize_input(
            valid_function, valid_args, valid_description))
        self.assertFalse(self.ai_function_caller.validate_and_sanitize_input(
            "", valid_args, valid_description))
        self.assertFalse(self.ai_function_caller.validate_and_sanitize_input(
            valid_function, [], valid_description))
        self.assertFalse(self.ai_function_caller.validate_and_sanitize_input(
            valid_function, valid_args, ""))

    @patch("src.refactoring_tool.ai_functions.openai.ChatCompletion.create")
    def test_call_ai_function(self, mock_openai_chat_completion_create: MagicMock):
        """
        Test the call_ai_function method of the AIFunctionCaller class.
        """
        mock_openai_chat_completion_create.return_value = MagicMock(
            choices=[MagicMock(message={"content": "Test AI response"})])

        function = "def test_function(a: str, b: str) -> str:"
        args = ["arg1", "arg2"]
        description = "Test description"

        ai_response = self.ai_function_caller.call_ai_function(
            function, args, description)
        self.assertEqual(ai_response, "Test AI response")

        with self.assertRaises(ValueError):
            self.ai_function_caller.call_ai_function("", args, description)


if __name__ == "__main__":
    unittest.main()
