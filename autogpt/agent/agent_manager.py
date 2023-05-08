"""Agent manager for managing GPT agents"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

from autogpt.config.config import Config, Singleton
from autogpt.llm_utils import create_chat_completion
from autogpt.types.openai import Message


class AgentManager(metaclass=Singleton):
    """Agent manager for managing GPT agents"""

    def __init__(self):
        self.next_key = 0
        self.agents: Dict[
            int, Tuple[str, List[Message], str]
        ] = {}  # key, (task, full_message_history, model)
        self.cfg = Config()

    def create_agent(self, task: str, prompt: str, model: str) -> tuple[int, str]:
        """Create a new agent and return the key and the agent's first reply
        Args:
            task: The task the agent is created for
            prompt: The prompt to start the conversation with
            model: The model to use for the agent"""
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
        """Message an agent and return the agent's reply
        Args:
            key: The key of the agent to message
            message: The message to send to the agent
        Returns:
            The agent's reply
        Raises:
            KeyError: If the key does not exist
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
        """List all agents"""
        return [(key, task) for key, (task, _, _) in self.agents.items()]

    def delete_agent(self, key: Union[str, int]) -> bool:
        """Delete an agent
        Args:
            key: The key of the agent to delete
        Returns:
            True if the agent was deleted, False if the agent did not exist
        """
        key = int(key)
        if key in self.agents:
            del self.agents[key]
            return True
        return False

    def _run_plugins_pre_instruction(self, messages: List[Message]) -> List[Message]:
        """Run all plugins that can handle pre-instruction
        Args:
            messages: The messages to run the plugins on
        Returns:
            The messages after the plugins have been run
        """
        for plugin in self.cfg.plugins:
            if plugin.can_handle_pre_instruction():
                messages.extend(plugin.pre_instruction(messages) or [])
        return messages

    def _generate_reply(self, model: str, messages: List[Message]) -> str:
        """Generate a reply from the model
        Args:
            model: The model to use
            messages: The messages to use as context
        Returns:
            The generated reply
        """
        return create_chat_completion(model=model, messages=messages)

    def _run_plugins_on_instruction(self, messages: List[Message]) -> List[Message]:
        """Run all plugins that can handle on-instruction
        Args:
            messages: The messages to run the plugins on
        Returns:
            The messages after the plugins have been run
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
        """Run all plugins that can handle post-instruction
        Args:
            agent_reply: The agent's reply to run the plugins on
        Returns:
            The agent's reply after the plugins have been run
        """
        for plugin in self.cfg.plugins:
            if plugin.can_handle_post_instruction():
                agent_reply = plugin.post_instruction(agent_reply)
        return agent_reply
