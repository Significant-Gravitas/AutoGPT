import unittest
import sys
from pathlib import Path
from unittest.mock import patch

from src.refactoring_tool.code_refactoring import *

class TestCodeRefactoring(unittest.TestCase):

    @patch('src.refactoring_tool.code_refactoring.CodeAnalysis.analyze_code_readability')
    @patch('src.refactoring_tool.code_refactoring.CodeAnalysis.analyze_code_performance')
    @patch('src.refactoring_tool.code_refactoring.CodeAnalysis.analyze_code_security')
    @patch('src.refactoring_tool.code_refactoring.CodeAnalysis.analyze_code_modularity')
    @patch('src.refactoring_tool.code_refactoring.CodeImprovement.refactor_variables')
    @patch('src.refactoring_tool.code_refactoring.CodeImprovement.optimize_loops')
    @patch('src.refactoring_tool.code_refactoring.CodeImprovement.simplify_conditionals')
    @patch('src.refactoring_tool.code_refactoring.CodeImprovement.refactor_functions')
    @patch('src.refactoring_tool.code_refactoring.TestGeneration.write_tests')
    @patch('src.refactoring_tool.code_refactoring.TestGeneration.write_integration_tests')
    @patch('src.refactoring_tool.code_refactoring.TestGeneration.write_system_tests')
    def test_refactor_code(
        self,
        mock_write_system_tests,
        mock_write_integration_tests,
        mock_write_tests,
        mock_refactor_functions,
        mock_simplify_conditionals,
        mock_optimize_loops,
        mock_refactor_variables,
        mock_analyze_code_modularity,
        mock_analyze_code_security,
        mock_analyze_code_performance,
        mock_analyze_code_readability,
    ):
        code = "def example_function(): pass"

        mock_analyze_code_readability.return_value = "Readable code example."
        mock_analyze_code_performance.return_value = ["Optimization suggestion 1", "Optimization suggestion 2"]
        mock_analyze_code_security.return_value = ["Security suggestion 1", "Security suggestion 2"]
        mock_analyze_code_modularity.return_value = ["Modularity suggestion 1", "Modularity suggestion 2"]

        mock_refactor_variables.return_value = "Refactored variables code."
        mock_optimize_loops.return_value = "Optimized loops code."
        mock_simplify_conditionals.return_value = "Simplified conditionals code."
        mock_refactor_functions.return_value = "Refactored functions code."

        mock_write_tests.return_value = "Unit tests code."
        mock_write_integration_tests.return_value = "Integration tests code."
        mock_write_system_tests.return_value = "System tests code."

        options = RefactoringOptions(analyze_security=True, generate_integration_tests=True)
        refactored_code, test_code = refactor_code(code, options)

        self.assertEqual(refactored_code, "Refactored functions code.")
        self.assertEqual(test_code, "Unit tests code.")

        mock_analyze_code_readability.assert_called_once()
        mock_analyze_code_performance.assert_called_once()
        mock_analyze_code_security.assert_called_once()
        mock_analyze_code_modularity.assert_called_once()
        mock_refactor_variables.assert_called_once()
        mock_optimize_loops.assert_called_once()
        mock_simplify_conditionals.assert_called_once()
        mock_refactor_functions.assert_called_once()
        mock_write_tests.assert_called_once()
        mock_write_integration_tests.assert_called_once()
        mock_write_system_tests.assert_called_once()


if __name__ == '__main__':
    unittest.main()
