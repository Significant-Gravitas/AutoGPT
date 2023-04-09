import unittest
from typing import Tuple
from src.refactoring_tool.code_refactoring import refactor_code, RefactoringOptions


class TestIntegration(unittest.TestCase):

    def test_code_refactoring(self):
        """
        Test the integration of various components in the code_refactoring.py module.
        """
        code = """
def foo(a, b):
    if a > b:
        return a
    else:
        return b
"""

        options = RefactoringOptions(
            analyze_readability=True,
            analyze_performance=True,
            analyze_security=False,
            analyze_modularity=True,
            refactor_variables=True,
            optimize_loops=True,
            simplify_conditionals=True,
            refactor_functions=True,
            generate_unit_tests=True,
            generate_integration_tests=False,
            generate_system_tests=False,
        )

        refactored_code, test_code = refactor_code(code, options)

        # Check if the refactored code is not empty
        self.assertIsNotNone(refactored_code)

        # Check if the generated test code is not empty
        self.assertIsNotNone(test_code)


if __name__ == "__main__":
    unittest.main()
