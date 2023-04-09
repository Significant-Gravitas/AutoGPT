import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, List
import openai

from src.agent_management.agent_manager import Agent, AgentManager



class TestAgent(unittest.TestCase):

    def setUp(self) -> None:
        self.agent = Agent()

    def test_init(self) -> None:
        self.assertIsInstance(self.agent, Agent)
        self.assertEqual(self.agent.model, "gpt-3.5-turbo")
        self.assertEqual(self.agent.message_history, [])

    def test_add_message(self) -> None:
        self.agent.add_message("system", "Hello, Assistant!")
        self.assertEqual(len(self.agent.message_history), 1)
        self.assertEqual(self.agent.message_history[0], {"role": "system", "content": "Hello, Assistant!"})

    def test_get_history(self) -> None:
        self.agent.add_message("system", "Hello, Assistant!")
        history = self.agent.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], {"role": "system", "content": "Hello, Assistant!"})


class TestAgentManager(unittest.TestCase):

    def setUp(self) -> None:
        self.api_key = "your_openai_api_key"
        self.agent_manager = AgentManager(self.api_key)

    def test_init(self) -> None:
        self.assertIsInstance(self.agent_manager, AgentManager)
        self.assertEqual(self.agent_manager.agents, {})

    def test_create_agent(self) -> None:
        self.agent_manager.create_agent("test_agent")
        self.assertIn("test_agent", self.agent_manager.agents)

    @patch('src.agent_management.agent_manager.openai.ChatCompletion.create')
    def test_send_message(self, mock_chat_completion_create) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].text.strip.return_value = "Hello, I am your Assistant!"
        mock_chat_completion_create.return_value = mock_response

        self.agent_manager.create_agent("test_agent")
        response = self.agent_manager.send_message("test_agent", "Hello, Assistant!")
        self.assertEqual(response, "Hello, I am your Assistant!")
        self.assertEqual(len(self.agent_manager.agents["test_agent"].message_history), 2)

    def test_list_agents(self) -> None:
        self.agent_manager.create_agent("test_agent")
        self.agent_manager.create_agent("test_agent2")
        agent_list = self.agent_manager.list_agents()
        self.assertIn("test_agent", agent_list)
        self.assertIn("test_agent2", agent_list)

    def test_delete_agent(self) -> None:
        self.agent_manager.create_agent("test_agent")
        self.assertIn("test_agent", self.agent_manager.agents)
        self.agent_manager.delete_agent("test_agent")
        self.assertNotIn("test_agent", self.agent_manager.agents)


if __name__ == "__main__":
    unittest.main()
