import unittest
from unittest.mock import MagicMock, patch
from src.refactoring_tool.ai_functions import call_ai_function
from src.refactoring_tool.code_improvements import CodeImprovement


class TestCodeImprovement(unittest.TestCase):
    def setUp(self) -> None:
        """
        Create a clean instance of the CodeImprovement class for each test case.
        """
        self.code_improvement = CodeImprovement()

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_refactor_variables(self, mock_call_ai_function: MagicMock):
        """
        Test the refactor_variables method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = "def add_numbers(num1, num2):\n    return num1 + num2"

        # Call the method
        suggestions = ["num1", "num2"]
        code = "def add_numbers(a, b):\n    return a + b"
        result = self.code_improvement.refactor_variables(suggestions, code)

        # Check the result
        self.assertEqual(
            result, "def add_numbers(num1, num2):\n    return num1 + num2")

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_optimize_loops(self, mock_call_ai_function: MagicMock):
        """
        Test the optimize_loops method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = "for i in range(5):\n    print(i)"

        # Call the method
        suggestions = ["use range instead of enumerate"]
        code = "for i, _ in enumerate(range(5)):\n    print(i)"
        result = self.code_improvement.optimize_loops(suggestions, code)

        # Check the result
        self.assertEqual(result, "for i in range(5):\n    print(i)")

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_simplify_conditionals(self, mock_call_ai_function: MagicMock):
        """
        Test the simplify_conditionals method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = "if x > 0:\n    print('Positive')"

        # Call the method
        suggestions = ["remove unnecessary else"]
        code = "if x > 0:\n    print('Positive')\nelse:\n    if x == 0:\n        pass"
        result = self.code_improvement.simplify_conditionals(suggestions, code)

        # Check the result
        self.assertEqual(result, "if x > 0:\n    print('Positive')")

    @patch("src.refactoring_tool.ai_functions.call_ai_function")
    def test_refactor_functions(self, mock_call_ai_function: MagicMock):
        """
        Test the refactor_functions method.
        """
        # Mock the AI function call response
        mock_call_ai_function.return_value = "def calculate_sum(a, b):\n    return a + b"

        # Call the method
        suggestions = ["rename function to calculate_sum"]
        code = "def add(a, b):\n    return a + b"
        result = self.code_improvement.refactor_functions(suggestions, code)

        # Check the result
        self.assertEqual(result, "def calculate_sum(a, b):\n    return a + b")


if __name__ == "__main__":
    unittest.main()
