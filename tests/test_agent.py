import unittest
from unittest.mock import patch, MagicMock
from autogpt.agent import Agent
from autogpt.utils import clean_input
from autogpt.speech import say_text

from autogpt.chat import chat_with_ai

from autogpt.config import Config


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.ai_name = "Test AI"
        self.memory = MagicMock()
        self.full_message_history = []
        self.next_action_count = 0
        self.command_registry = MagicMock()
        self.config = Config()
        self.system_prompt = "System prompt"
        self.triggering_prompt = "Triggering prompt"

        self.agent = Agent(
            self.ai_name,
            self.memory,
            self.full_message_history,
            self.next_action_count,
            self.command_registry,
            self.config,
            self.system_prompt,
            self.triggering_prompt,
        )

    def test_agent_initialization(self):
        self.assertEqual(self.agent.ai_name, self.ai_name)
        self.assertEqual(self.agent.memory, self.memory)
        self.assertEqual(self.agent.full_message_history, self.full_message_history)
        self.assertEqual(self.agent.next_action_count, self.next_action_count)
        self.assertEqual(self.agent.command_registry, self.command_registry)
        self.assertEqual(self.agent.config, self.config)
        self.assertEqual(self.agent.system_prompt, self.system_prompt)
        self.assertEqual(self.agent.triggering_prompt, self.triggering_prompt)

    # More test methods can be added for specific agent interactions
    # For example, mocking chat_with_ai and testing the agent's interaction loop
    

    @patch("autogpt.chat.chat_with_ai")
    @patch("autogpt.app.get_command")
    @patch("autogpt.app.execute_command")
    @patch("autogpt.speech.say_text")
    @patch("autogpt.utils.clean_input")
    def test_agent_interaction_loop(
        self, mock_clean_input, mock_say_text, mock_execute_command, mock_get_command, mock_chat_with_ai
    ):
        # Mock chat_with_ai to return a predefined response
        mock_chat_with_ai.return_value = '{"action": {"command": "test_command", "arguments": "test_args"}}'

        # Mock get_command to return a predefined command and arguments
        mock_get_command.return_value = ("test_command", "test_args")

        # Mock execute_command to return a predefined result
        mock_execute_command.return_value = "Test result"

        # Mock say_text to do nothing
        mock_say_text.return_value = None

        # Mock clean_input to return 'n' (exit)
        mock_clean_input.return_value = "n"

        self.agent.start_interaction_loop()

        # Check if chat_with_ai was called
        mock_chat_with_ai.assert_called()

        # Check if get_command was called with the expected JSON response
        mock_get_command.assert_called_with({"action": {"command": "test_command", "arguments": "test_args"}})

        # Check if execute_command was called with the expected command and arguments
        mock_execute_command.assert_called_with(
            self.agent.command_registry,
            "test_command",
            "test_args",
            "gpt-3.5-turbo",
        )

        # Check if say_text was called with the expected text
        mock_say_text.assert_called_with("I want to execute test_command")

        # Check if clean_input was called
        mock_clean_input.assert_called()

if __name__ == '__main__':
    unittest.main()
