import unittest
from unittest.mock import MagicMock, patch
from src.agent_management.agent_manager import Agent, AgentManager
from src.refactoring_tool.ai_functions import call_ai_function, validate_and_sanitize_input
from src.refactoring_tool.code_analysis import CodeAnalysis
from src.refactoring_tool.code_improvement import CodeImprovement
from src.refactoring_tool.code_refactoring import refactor_code, RefactoringOptions
from src.refactoring_tool.test_generation import TestGeneration


class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.agent_name = "test_agent"

    def test_ai_function_call(self):
        with patch("src.refactoring_tool.ai_functions.call_ai_function") as mock_call_ai_function:
            mock_call_ai_function.return_value = "12"

            function = "def calculate_sum(a: int, b: int) -> int:"
            args = [5, 7]
            description = "Calculate the sum of a and b and return the result."
            result = call_ai_function(function, tuple(map(str, args)), description)  # Convert args to strings

            self.assertEqual(result, "12")

    def test_code_analysis(self):
        code = "def example_function(a, b):\n    return a + b"
        readability_result = CodeAnalysis.analyze_code_readability(code)
        self.assertIsNotNone(readability_result)

    def test_code_improvement(self):
        suggestions = ["Use more descriptive variable names."]
        code = "def example_function(a, b):\n    return a + b"
        improved_code = CodeImprovement.refactor_variables(suggestions, code)
        self.assertIsNotNone(improved_code)

    def test_code_refactoring(self):
        code = "def example_function(a, b):\n    return a + b"
        options = RefactoringOptions()
        refactored_code, test_code = refactor_code(code, options)
        self.assertIsNotNone(refactored_code)
        self.assertIsNotNone(test_code)

    def test_test_generation(self):
        code = "def example_function(a, b):\n    return a + b"
        unit_test_code = TestGeneration.write_tests(code)
        self.assertIsNotNone(unit_test_code)

    @patch("openai.Completion.create")  
    def test_agent_manager(self, mock_completion_create):
        mock_completion_create.return_value = MagicMock(choices=[MagicMock(text="I am fine. Thank you!")])

        agent_manager = AgentManager(api_key="dummy_key")
        agent_manager.create_agent(self.agent_name)
        
        message = "Hello, how are you?"
        response = agent_manager.send_message(self.agent_name, message)
        self.assertIsNotNone(response)
        
        # Delete the agent and assert that it cannot be accessed
        agent_manager.delete_agent(self.agent_name)
        with self.assertRaises(ValueError):
            agent_manager.send_message(self.agent_name, message)
        
        # Recreate the agent for tearDown
        agent_manager.create_agent(self.agent_name)


if __name__ == "__main__":
    unittest.main()
