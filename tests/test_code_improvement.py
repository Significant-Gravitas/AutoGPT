import unittest
from unittest.mock import patch
from typing import List
from src.refactoring_tool.code_improvement import CodeImprovement


class TestCodeImprovement(unittest.TestCase):

    @patch('src.refactoring_tool.code_improvement.call_ai_function')
    def test_refactor_variables(self, mock_call_ai_function):
        suggestions = ["rename 'x' to 'count'", "rename 'y' to 'total'"]
        code = "def example_function(x, y):\n    return x + y"

        mock_call_ai_function.return_value = "def example_function(count, total):\n    return count + total"

        result = CodeImprovement.refactor_variables(suggestions, code)
        self.assertEqual(result, "def example_function(count, total):\n    return count + total")
        mock_call_ai_function.assert_called_once()

    @patch('src.refactoring_tool.code_improvement.call_ai_function')
    def test_optimize_loops(self, mock_call_ai_function):
        suggestions = ["use enumerate instead of range(len)"]
        code = "for i in range(len(items)):\n    print(i, items[i])"

        mock_call_ai_function.return_value = "for i, item in enumerate(items):\n    print(i, item)"

        result = CodeImprovement.optimize_loops(suggestions, code)
        self.assertEqual(result, "for i, item in enumerate(items):\n    print(i, item)")
        mock_call_ai_function.assert_called_once()

    @patch('src.refactoring_tool.code_improvement.call_ai_function')
    def test_simplify_conditionals(self, mock_call_ai_function):
        suggestions = ["replace 'if x == True:' with 'if x:'"]
        code = "if x == True:\n    print('x is True')"

        mock_call_ai_function.return_value = "if x:\n    print('x is True')"

        result = CodeImprovement.simplify_conditionals(suggestions, code)
        self.assertEqual(result, "if x:\n    print('x is True')")
        mock_call_ai_function.assert_called_once()

    @patch('src.refactoring_tool.code_improvement.call_ai_function')
    def test_refactor_functions(self, mock_call_ai_function):
        suggestions = ["rename 'example_function' to 'renamed_example_function'"]
        code = "def example_function():\n    pass"

        mock_call_ai_function.return_value = "def renamed_example_function():\n    pass"

        result = CodeImprovement.refactor_functions(suggestions, code)
        self.assertEqual(result, "def renamed_example_function():\n    pass")
        mock_call_ai_function.assert_called_once()


if __name__ == '__main__':
    unittest.main()
