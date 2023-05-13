from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agent import Agent

from autogpt.config import Config
from autogpt.json_utils.utilities import (
    LLM_DEFAULT_RESPONSE_FORMAT,
    is_string_valid_json,
)
from autogpt.llm.base import Message
from autogpt.llm.llm_utils import create_chat_completion
from autogpt.log_cycle.log_cycle import PROMPT_SUMMARY_FILE_NAME, SUMMARY_FILE_NAME
from autogpt.logs import logger
from autogpt.memory.context import ContextMemory


class MessageHistory:
    agent: Agent

    messages: list[Message]
    summary: str

    last_memory_index: int

    def __init__(self, agent: Agent):
        self.agent = agent
        self.messages = []
        self.summary = "I was created."
        self.last_memory_index = 0

    def add(self, message: Message):
        self.messages.append(message)

    def get_trimmed_messages(
        self,
        current_message_chain: list[Message],
    ) -> list[Message]:
        """
        Returns a list of trimmed messages: messages which are in the message history
        but not in current_message_chain.

        Args:
            current_message_chain (list): The messages currently in the context.

        Returns:
            list: A list of messages that are in full_message_history with an index higher than last_memory_index and absent from current_context.
        """
        # Select messages in full_message_history with an index higher than last_memory_index
        new_messages = [
            msg for i, msg in enumerate(self.messages) if i > self.last_memory_index
        ]

        # Remove messages that are already present in current_message_chain
        new_messages_not_in_chain = [
            msg for msg in new_messages if msg not in current_message_chain
        ]

        # Find the index of the last message processed
        if new_messages_not_in_chain:
            last_message = new_messages_not_in_chain[-1]
            self.last_memory_index = self.messages.index(last_message)

        return new_messages_not_in_chain

    def save_memory_trimmed_from_context_window(
        self,
        next_message_to_add_index: int,
        permanent_memory: ContextMemory,
    ):
        while next_message_to_add_index >= 0:
            message_content = self.messages[next_message_to_add_index]["content"]
            if is_string_valid_json(message_content, LLM_DEFAULT_RESPONSE_FORMAT):
                next_message = self.messages[next_message_to_add_index + 1]
                memory_to_add = self.format_memory(
                    message_content, next_message["content"]
                )
                logger.debug(f"Storing the following memory: {memory_to_add}")
                permanent_memory.add(memory_to_add)

            next_message_to_add_index -= 1

    def format_memory(self, assistant_reply: str, next_message_content: str):
        # the next_message_content is a variable to stores either the user_input or the command following the assistant_reply
        result = (
            "None"
            if next_message_content.startswith("Command")
            else next_message_content
        )
        user_input = (
            "None"
            if next_message_content.startswith("Human feedback")
            else next_message_content
        )

        return f"Assistant Reply: {assistant_reply}\nResult: {result}\nHuman Feedback: {user_input}"

    def summary_message(self) -> Message:
        return {
            "role": "system",
            "content": f"This reminds you of these events from your past: \n{self.summary}",
        }

    def update_running_summary(self, new_events: list[Message]) -> Message:
        """
        This function takes a list of dictionaries representing new events and combines them with the current summary,
        focusing on key and potentially important information to remember. The updated summary is returned in a message
        formatted in the 1st person past tense.

        Args:
            new_events (List[Dict]): A list of dictionaries containing the latest events to be added to the summary.

        Returns:
            str: A message containing the updated summary of actions, formatted in the 1st person past tense.

        Example:
            new_events = [{"event": "entered the kitchen."}, {"event": "found a scrawled note with the number 7"}]
            update_running_summary(new_events)
            # Returns: "This reminds you of these events from your past: \nI entered the kitchen and found a scrawled note saying 7."
        """
        cfg = Config()

        if not new_events:
            return self.summary_message()

        # Create a copy of the new_events list to prevent modifying the original list
        new_events = copy.deepcopy(new_events)

        # Replace "assistant" with "you". This produces much better first person past tense results.
        for event in new_events:
            if event["role"].lower() == "assistant":
                event["role"] = "you"

                # Remove "thoughts" dictionary from "content"
                content_dict = json.loads(event["content"])
                if "thoughts" in content_dict:
                    del content_dict["thoughts"]
                event["content"] = json.dumps(content_dict)

            elif event["role"].lower() == "system":
                event["role"] = "your computer"

            # Delete all user messages
            elif event["role"] == "user":
                new_events.remove(event)

        prompt = f'''Your task is to create a concise running summary of actions and information results in the provided text, focusing on key and potentially important information to remember.

    You will receive the current summary and the your latest actions. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.

    Summary So Far:
    """
    {self.summary}
    """

    Latest Development:
    """
    {new_events or "Nothing new happened."}
    """
    '''

        messages: list[Message] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        self.agent.log_cycle_handler.log_cycle(
            self.agent.config.ai_name,
            self.agent.created_at,
            self.agent.cycle_count,
            messages,
            PROMPT_SUMMARY_FILE_NAME,
        )

        self.summary = create_chat_completion(messages, cfg.fast_llm_model)

        self.agent.log_cycle_handler.log_cycle(
            self.agent.config.ai_name,
            self.agent.created_at,
            self.agent.cycle_count,
            self.summary,
            SUMMARY_FILE_NAME,
        )

        return self.summary_message()
