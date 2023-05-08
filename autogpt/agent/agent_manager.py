from __future__ import annotations

from typing import Dict, List, Tuple, Union

from autogpt.config.config import Config
from autogpt.llm import Message, create_chat_completion
from autogpt.singleton import Singleton


class AgentManager(metaclass=Singleton):
    """Agent manager for managing GPT agents"""

    def __init__(self):
        """Initialize the agent manager"""
        self.next_key = 0
        self.agents: Dict[
            int, Tuple[str, List[Message], str]
        ] = {}  # key, (task, full_message_history, model)
        self.cfg = Config()

    def create_agent(self, task: str, prompt: str, model: str) -> tuple[int, str]:
        """
        Create a new agent with the given task, prompt and model.
        Args:
            task: The task of the agent
            prompt: The prompt message to the agent
            model: The model to use for the agent

        Returns:
            The key of the agent and the first reply from the agent to the prompt message provided by the user to the agent when it was created.
        """
        messages: List[Message] = [{"role": "user", "content": prompt}]
        messages = self._run_plugins_pre_instruction(messages)
        agent_reply = self._generate_reply(model, messages)
        messages.append({"role": "assistant", "content": agent_reply})
        messages = self._run_plugins_on_instruction(messages)
        key = self.next_key
        self.next_key += 1
        self.agents[key] = (task, messages, model)
        agent_reply = self._run_plugins_post_instruction(agent_reply)
        return key, agent_reply

    def message_agent(self, key: Union[str, int], message: str) -> str:
        """
        Send a message to an agent with the given key.
        Args:
            key: The key of the agent
            message: The message to send to the agent

        Returns:
            The reply from the agent to the message provided by the user.
        """
        task, messages, model = self.agents[int(key)]
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
        """
        Delete an agent with the given key.
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
