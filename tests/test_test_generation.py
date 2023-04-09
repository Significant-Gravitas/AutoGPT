import unittest
from unittest.mock import MagicMock, patch
from typing import List, Optional
from src.refactoring_tool.ai_functions import call_ai_function
from src.refactoring_tool.test_generation import TestGeneration


class TestTestGeneration(unittest.TestCase):
    def setUp(self) -> None:
        """
        Create a clean instance of the TestGeneration class for each test case.
        """
        self.test_generation = TestGeneration()

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_write_tests(self, mock_call_ai_function: MagicMock):
        """
        Test the write_tests method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = "def test_add():\n    assert add(1, 2) == 3"

        # Call the method
        code = "def add(a, b):\n    return a + b"
        focus: Optional[List[str]] = None
        result = self.test_generation.write_tests(code, focus)

        # Check the result
        self.assertEqual(result, "def test_add():\n    assert add(1, 2) == 3")

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_write_integration_tests(self, mock_call_ai_function: MagicMock):
        """
        Test the write_integration_tests method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = "def test_integration():\n    assert integration_function() == True"

        # Call the method
        code = "def integration_function():\n    return True"
        focus: Optional[List[str]] = None
        result = self.test_generation.write_integration_tests(code, focus)

        # Check the result
        self.assertEqual(
            result, "def test_integration():\n    assert integration_function() == True")

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_write_system_tests(self, mock_call_ai_function: MagicMock):
        """
        Test the write_system_tests method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = "def test_system():\n    assert system_function() == 'System Working'"

        # Call the method
        code = "def system_function():\n    return 'System Working'"
        focus: Optional[List[str]] = None
        result = self.test_generation.write_system_tests(code, focus)

        # Check the result
        self.assertEqual(
            result, "def test_system():\n    assert system_function() == 'System Working'")


if __name__ == "__main__":
    unittest.main()
