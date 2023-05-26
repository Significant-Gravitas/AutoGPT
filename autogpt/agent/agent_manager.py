"""Agent manager for managing GPT agents"""
from __future__ import annotations

from autogpt.config import Config
from autogpt.llm.base import ChatSequence
from autogpt.llm.chat import Message, create_chat_completion
from autogpt.singleton import Singleton


class AgentManager(metaclass=Singleton):
    """Agent manager for managing GPT agents"""

    def __init__(self):
        self.next_key = 0
        self.agents: dict[
            int, tuple[str, list[Message], str]
        ] = {}  # key, (task, full_message_history, model)
        self.cfg = Config()

    # Create new GPT agent
    # TODO: Centralise use of create_chat_completion() to globally enforce token limit

    def create_agent(
        self, task: str, creation_prompt: str, model: str
    ) -> tuple[int, str]:
        """Create a new agent and return its key

        Args:
            task: The task to perform
            creation_prompt: Prompt passed to the LLM at creation
            model: The model to use to run this agent

        Returns:
            The key of the new agent
        """
        messages = ChatSequence.for_model(model, [Message("user", creation_prompt)])

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_pre_instruction():
                continue
            if plugin_messages := plugin.pre_instruction(messages.raw()):
                messages.extend([Message(**raw_msg) for raw_msg in plugin_messages])
        # Start GPT instance
        agent_reply = create_chat_completion(prompt=messages)

        messages.add("assistant", agent_reply)

        plugins_reply = ""
        for i, plugin in enumerate(self.cfg.plugins):
            if not plugin.can_handle_on_instruction():
                continue
            if plugin_result := plugin.on_instruction([m.raw() for m in messages]):
                sep = "\n" if i else ""
                plugins_reply = f"{plugins_reply}{sep}{plugin_result}"

        if plugins_reply and plugins_reply != "":
            messages.add("assistant", plugins_reply)
        key = self.next_key
        # This is done instead of len(agents) to make keys unique even if agents
        # are deleted
        self.next_key += 1

        self.agents[key] = (task, list(messages), model)

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_post_instruction():
                continue
            agent_reply = plugin.post_instruction(agent_reply)

        return key, agent_reply

    def message_agent(self, key: str | int, message: str) -> str:
        """Send a message to an agent and return its response

        Args:
            key: The key of the agent to message
            message: The message to send to the agent

        Returns:
            The agent's response
        """
        task, messages, model = self.agents[int(key)]

        # Add user message to message history before sending to agent
        messages = ChatSequence.for_model(model, messages)
        messages.add("user", message)

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_pre_instruction():
                continue
            if plugin_messages := plugin.pre_instruction([m.raw() for m in messages]):
                messages.extend([Message(**raw_msg) for raw_msg in plugin_messages])

        # Start GPT instance
        agent_reply = create_chat_completion(prompt=messages)

        messages.add("assistant", agent_reply)

        plugins_reply = agent_reply
        for i, plugin in enumerate(self.cfg.plugins):
            if not plugin.can_handle_on_instruction():
                continue
            if plugin_result := plugin.on_instruction([m.raw() for m in messages]):
                sep = "\n" if i else ""
                plugins_reply = f"{plugins_reply}{sep}{plugin_result}"
        # Update full message history
        if plugins_reply and plugins_reply != "":
            messages.add("assistant", plugins_reply)

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_post_instruction():
                continue
            agent_reply = plugin.post_instruction(agent_reply)

        return agent_reply

    def list_agents(self) -> list[tuple[str | int, str]]:
        """Return a list of all agents

        Returns:
            A list of tuples of the form (key, task)
        """

        # Return a list of agent keys and their tasks
        return [(key, task) for key, (task, _, _) in self.agents.items()]

    def delete_agent(self, key: str | int) -> bool:
        """Delete an agent from the agent manager

        Args:
            key: The key of the agent to delete

        Returns:
            True if successful, False otherwise
        """

        try:
            del self.agents[int(key)]
            return True
        except KeyError:
            return False
