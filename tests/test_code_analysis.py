import unittest
from unittest.mock import patch
from typing import List
from src.refactoring_tool.code_analysis import CodeAnalysis


class TestCodeAnalysis(unittest.TestCase):

    @patch('src.refactoring_tool.code_analysis.call_ai_function')
    def test_analyze_code_readability(self, mock_call_ai_function):
        code = "def example_function(): pass"

        mock_call_ai_function.return_value = "Readable code example."

        result = CodeAnalysis.analyze_code_readability(code)
        self.assertEqual(result, "Readable code example.")
        mock_call_ai_function.assert_called_once()

    @patch('src.refactoring_tool.code_analysis.call_ai_function')
    def test_analyze_code_performance(self, mock_call_ai_function):
        code = "def example_function(): pass"

        mock_call_ai_function.return_value = '["Optimization suggestion 1", "Optimization suggestion 2"]'

        result = CodeAnalysis.analyze_code_performance(code)
        self.assertEqual(result, ["Optimization suggestion 1", "Optimization suggestion 2"])
        mock_call_ai_function.assert_called_once()

    @patch('src.refactoring_tool.code_analysis.call_ai_function')
    def test_analyze_code_security(self, mock_call_ai_function):
        code = "def example_function(): pass"

        mock_call_ai_function.return_value = '["Security suggestion 1", "Security suggestion 2"]'

        result = CodeAnalysis.analyze_code_security(code)
        self.assertEqual(result, ["Security suggestion 1", "Security suggestion 2"])
        mock_call_ai_function.assert_called_once()

    @patch('src.refactoring_tool.code_analysis.call_ai_function')
    def test_analyze_code_modularity(self, mock_call_ai_function):
        code = "def example_function(): pass"

        mock_call_ai_function.return_value = '["Modularity suggestion 1", "Modularity suggestion 2"]'

        result = CodeAnalysis.analyze_code_modularity(code)
        self.assertEqual(result, ["Modularity suggestion 1", "Modularity suggestion 2"])
        mock_call_ai_function.assert_called_once()


if __name__ == '__main__':
    unittest.main()
