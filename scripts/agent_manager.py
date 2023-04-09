from typing import Dict, List
import openai


class Agent:
    """
    The Agent class represents an individual chat agent that interacts with OpenAI's GPT model.
    It manages the agent's conversation history and GPT model selection.
    """

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        """
        Initializes an Agent instance with a specified model.

        :param model: The name of the GPT model to be used by the agent (default: "gpt-3.5-turbo").
        """
        self.model = model
        self.message_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the agent's conversation history.

        :param role: The role of the sender in the conversation ("system" or "assistant").
        :param content: The content of the message.
        """
        self.message_history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """
        Returns the agent's conversation history.

        :return: A list of dictionaries representing the conversation history.
        """
        return self.message_history


class AgentManager:
    """
    The AgentManager class is designed to manage multiple instances of the Agent class.
    It provides methods for creating agents, sending messages to agents, listing agents, and deleting agents.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initializes an AgentManager instance with a specified API key.

        :param api_key: The OpenAI API key.
        """
        openai.api_key = api_key
        self.agents: Dict[str, Agent] = {}

    def create_agent(self, agent_name: str, model: str = "gpt-3.5-turbo") -> None:
        """
        Creates a new agent and stores it in the agents dictionary.

        :param agent_name: A unique name for the agent.
        :param model: The name of the GPT model to be used by the agent (default: "gpt-3.5-turbo").
        """
        self.agents[agent_name] = Agent(model)

    def send_message(self, agent_name: str, message: str) -> str:
        """
        Sends a message to the specified agent and returns its response.

        :param agent_name: The name of the agent to send the message to.
        :param message: The message to be sent to the agent.
        :return: The agent's response.
        :raises ValueError: If the agent is not found.
        """
        agent = self.agents.get(agent_name)
        if agent is not None:
            agent.add_message("system", message)
            response = openai.ChatCompletion.create(
                model=agent.model,
                messages=agent.get_history(),
                max_tokens=150
            )
            agent_response = response.choices[0].text.strip()
            agent.add_message("assistant", agent_response)
            return agent_response
        else:
            raise ValueError(f"Agent '{agent_name}' not found.")

    def list_agents(self) -> List[str]:
        """
        Returns a list of agent names.

        :return: A list of agent names.
        """
        return list(self.agents.keys())

    def delete_agent(self, agent_name: str) -> None:
        """
        Deletes an agent from the agents dictionary.

        :param agent_name: The name of the agent to be deleted.
        :raises ValueError: If the agent is not found.
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
        else:
            raise ValueError(f"Agent '{agent_name}' not found.")
