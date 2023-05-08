"""Agent manager for managing GPT agents"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

from autogpt.config.config import Config
from autogpt.llm import Message, create_chat_completion
from autogpt.singleton import Singleton


class AgentManager(metaclass=Singleton):
    """Agent manager for managing GPT agents"""

    def __init__(self):
        self.next_key = 0
        self.agents: Dict[
            int, Tuple[str, List[Message], str]
        ] = {}  # key, (task, full_message_history, model)
        self.cfg = Config()

    # Create new GPT agent
    # TODO: Centralise use of create_chat_completion() to globally enforce token limit

    def create_agent(self, task: str, prompt: str, model: str) -> Tuple[int, str]:
        """Create a new agent and return its key

        Args:
            task: The task to perform
            prompt: The prompt to use
            model: The model to use

        Returns:
            The key of the new agent
        """
        messages: List[Message] = [{"role": "user", "content": prompt}]
        messages = self._run_plugins_pre_instruction(messages)
        agent_reply = self._generate_reply(model, messages)
        messages.append({"role": "assistant", "content": agent_reply})
        messages = self._run_plugins_on_instruction(messages)
        key = self.next_key
        # This is done instead of len(agents) to make keys unique even if agents
        # are deleted
        self.next_key += 1

        self.agents[key] = (task, messages, model)

        agent_reply = self._run_plugins_post_instruction(agent_reply)
        return key, agent_reply

    def message_agent(self, key: Union[str, int], message: str) -> str:
        """Send a message to an agent and return its response

        Args:
            key: The key of the agent to message
            message: The message to send to the agent

        Returns:
            The agent's response
        """
        task, messages, model = self.agents[int(key)]

        # Add user message to message history before sending to agent
        messages.append({"role": "user", "content": message})
        messages = self._run_plugins_pre_instruction(messages)
        agent_reply = self._generate_reply(model, messages)
        messages.append({"role": "assistant", "content": agent_reply})

        messages = self._run_plugins_on_instruction(messages)
        agent_reply = self._run_plugins_post_instruction(agent_reply)
        return agent_reply

    def list_agents(self) -> list[tuple[Union[str, int], str]]:
        """
        List all agents and their tasks.
        Returns:
            A list of tuples containing the key and task of each agent.
        """
        return [(key, task) for key, (task, _, _) in self.agents.items()]

    def delete_agent(self, key: Union[str, int]) -> bool:
        """Delete an agent from the agent manager
        Args:
            key: The key of the agent to delete.

        Returns:
            True if the agent was deleted, False if the agent was not found.
        """
        key = int(key)
        if key in self.agents:
            del self.agents[key]
            return True
        return False

    def _run_plugins_pre_instruction(self, messages: List[Message]) -> List[Message]:
        """
        Run plugins that can handle pre instruction events.
        Args:
            messages: The messages to send to the plugins.

        Returns:
            The messages to send to the model.
        """
        for plugin in self.cfg.plugins:
            if plugin.can_handle_pre_instruction():
                messages.extend(plugin.pre_instruction(messages) or [])
        return messages

    def _generate_reply(self, model: str, messages: List[Message]) -> str:
        """
        Generate a reply from the given model and messages.
        Args:
            model: The model to use to generate the reply.
            messages: The messages to send to the model.

        Returns:
            The reply from the model.
        """
        return create_chat_completion(model=model, messages=messages)

    def _run_plugins_on_instruction(self, messages: List[Message]) -> List[Message]:
        """
        Run plugins that can handle on instruction events.
        Args:
            messages: The messages to send to the plugins.

        Returns:
            The messages to send to the user.
        """
        reply_added = False
        for plugin in self.cfg.plugins:
            if plugin.can_handle_on_instruction():
                plugin_result = plugin.on_instruction(messages)
                if plugin_result:
                    if not reply_added:
                        messages.append({"role": "assistant", "content": plugin_result})
                        reply_added = True
                    else:
                        messages[-1]["content"] += f"\n{plugin_result}"
        return messages

    def _run_plugins_post_instruction(self, agent_reply: str) -> str:
        """
        Run plugins that can handle post instruction events.
        Args:
            agent_reply: The reply from the agent to the user.

        Returns:
            The reply from the agent to the user.
        """
        for plugin in self.cfg.plugins:
            if plugin.can_handle_post_instruction():
                agent_reply = plugin.post_instruction(agent_reply)
        return agent_reply
