import unittest
from unittest.mock import MagicMock, patch

import autogpt.agent.agent_manager as agent_manager
from autogpt.app import execute_command, list_agents, start_agent


class TestCommands(unittest.TestCase):
    def test_make_agent(self):
        with patch("openai.ChatCompletion.create") as mock:
            obj = MagicMock()
            obj.response.choices[0].messages[0].content = "Test message"
            mock.return_value = obj
            start_agent("Test Agent", "chat", "Hello, how are you?", "gpt2")
            agents = list_agents()
            self.assertEqual("List of agents:\n0: chat", agents)
            start_agent("Test Agent 2", "write", "Hello, how are you?", "gpt2")
            agents = list_agents()
            self.assertEqual("List of agents:\n0: chat\n1: write", agents)
