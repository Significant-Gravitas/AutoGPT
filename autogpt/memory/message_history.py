from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from autogpt.agent import Agent

from autogpt.config import Config
from autogpt.json_utils.utilities import (
    LLM_DEFAULT_RESPONSE_FORMAT,
    is_string_valid_json,
)
from autogpt.llm.base import Message
from autogpt.llm.utils import create_chat_completion
from autogpt.log_cycle.log_cycle import PROMPT_SUMMARY_FILE_NAME, SUMMARY_FILE_NAME
from autogpt.logs import logger


@dataclass
class MessageHistory:
    agent: Agent

    messages: list[Message] = field(default_factory=list)
    summary: str = "I was created"

    last_trimmed_index: int = 0

    def __getitem__(self, i: int):
        return self.messages[i]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def add(self, role: Literal["system", "user", "assistant"], content: str):
        return self.append({"role": role, "content": content})

    def append(self, message: Message):
        return self.messages.append(message)

    def trim_messages(
        self,
        current_message_chain: list[Message],
    ) -> tuple[Message, list[Message]]:
        """
        Returns a list of trimmed messages: messages which are in the message history
        but not in current_message_chain.

        Args:
            current_message_chain (list[Message]): The messages currently in the context.

        Returns:
            Message: A message with the new running summary after adding the trimmed messages.
            list[Message]: A list of messages that are in full_message_history with an index higher than last_trimmed_index and absent from current_message_chain.
        """
        # Select messages in full_message_history with an index higher than last_trimmed_index
        new_messages = [
            msg for i, msg in enumerate(self) if i > self.last_trimmed_index
        ]

        # Remove messages that are already present in current_message_chain
        new_messages_not_in_chain = [
            msg for msg in new_messages if msg not in current_message_chain
        ]

        # agent.history.archive(
        #     new_messages_not_in_chain,
        #     agent.memory,
        # )

        new_summary_message = self.update_running_summary(
            new_events=new_messages_not_in_chain
        )

        # Find the index of the last message processed
        if new_messages_not_in_chain:
            last_message = new_messages_not_in_chain[-1]
            self.last_trimmed_index = self.messages.index(last_message)

        return new_summary_message, new_messages_not_in_chain

    @staticmethod
    def filter_message_pairs(messages: list[Message]):
        """
        Yields:
            Message: a message from the AI containing a proposed action
            Message: the message containing the result of the AI's proposed action
        """
        i = 0
        while i < len(messages):
            ai_message = messages[i]
            if is_string_valid_json(ai_message["content"], LLM_DEFAULT_RESPONSE_FORMAT):
                result_message = messages[i + 1]
                yield ai_message, result_message
            i += 1

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
                try:
                    content_dict = json.loads(event["content"])
                    if "thoughts" in content_dict:
                        del content_dict["thoughts"]
                    event["content"] = json.dumps(content_dict)
                except json.decoder.JSONDecodeError:
                    if cfg.debug_mode:
                        logger.error(f"Error: Invalid JSON: {event['content']}\n")

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
