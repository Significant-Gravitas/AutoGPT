import unittest
from src.agent_manager import Agent, AgentManager

class TestAgent(unittest.TestCase):

    def test_agent_creation(self):
        agent = Agent()
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.model, "gpt-3.5-turbo")
        self.assertEqual(agent.message_history, [])

    def test_add_message(self):
        agent = Agent()
        agent.add_message("system", "Test message")
        self.assertEqual(len(agent.message_history), 1)
        self.assertEqual(agent.message_history[0], {"role": "system", "content": "Test message"})

    def test_get_history(self):
        agent = Agent()
        agent.add_message("system", "Test message")
        history = agent.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], {"role": "system", "content": "Test message"})


class TestAgentManager(unittest.TestCase):

    def setUp(self):
        self.manager = AgentManager(api_key="your_openai_api_key")

    def test_create_agent(self):
        self.manager.create_agent("test_agent")
        self.assertIn("test_agent", self.manager.agents)
        self.assertIsInstance(self.manager.agents["test_agent"], Agent)

    def test_list_agents(self):
        self.manager.create_agent("test_agent")
        agent_list = self.manager.list_agents()
        self.assertEqual(agent_list, ["test_agent"])

    def test_delete_agent(self):
        self.manager.create_agent("test_agent")
        self.manager.delete_agent("test_agent")
        self.assertNotIn("test_agent", self.manager.agents)

    def test_delete_agent_not_found(self):
        with self.assertRaises(ValueError):
            self.manager.delete_agent("non_existent_agent")


# Make sure to replace "your_openai_api_key" with your actual OpenAI API key.
# Note that running this test will make API calls, so you may want to mock the API calls for testing purposes.
# We have not mocked the `send_message` method as it requires mocking the OpenAI API. You can do that using the `unittest.mock` library if needed.

if __name__ == "__main__":
    unittest.main()
