import unittest
from unittest.mock import MagicMock, patch
from src.refactoring_tool.ai_functions import call_ai_function
from src.refactoring_tool.code_analysis import CodeAnalysis


class TestCodeAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        """
        Create a clean instance of the CodeAnalysis class for each test case.
        """
        self.code_analysis = CodeAnalysis()

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_analyze_code_readability(self, mock_call_ai_function: MagicMock):
        """
        Test the analyze_code_readability method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = "The code is easy to read."

        # Call the method
        code = "def add(a, b):\n    return a + b"
        result = self.code_analysis.analyze_code_readability(code)

        # Check the result
        self.assertEqual(result, "The code is easy to read.")

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_analyze_code_performance(self, mock_call_ai_function: MagicMock):
        """
        Test the analyze_code_performance method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = '["Use list comprehensions.", "Optimize loop conditions."]'

        # Call the method
        code = "def add(a, b):\n    return a + b"
        result = self.code_analysis.analyze_code_performance(code)

        # Check the result
        self.assertEqual(
            result, ["Use list comprehensions.", "Optimize loop conditions."])

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_analyze_code_security(self, mock_call_ai_function: MagicMock):
        """
        Test the analyze_code_security method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = '["Sanitize user inputs.", "Use secure hashing algorithms."]'

        # Call the method
        code = "def add(a, b):\n    return a + b"
        result = self.code_analysis.analyze_code_security(code)

        # Check the result
        self.assertEqual(
            result, ["Sanitize user inputs.", "Use secure hashing algorithms."])

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_analyze_code_modularity(self, mock_call_ai_function: MagicMock):
        """
        Test the analyze_code_modularity method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = '["Break down large functions.", "Use classes and modules."]'

        # Call the method
        code = "def add(a, b):\n    return a + b"
        result = self.code_analysis.analyze_code_modularity(code)

        # Check the result
        self.assertEqual(
            result, ["Break down large functions.", "Use classes and modules."])


if __name__ == "__main__":
    unittest.main()
