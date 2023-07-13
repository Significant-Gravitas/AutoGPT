from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from autogpt.agents import Agent, BaseAgent
    from autogpt.config import Config

from autogpt.json_utils.utilities import extract_dict_from_response
from autogpt.llm.base import ChatSequence, Message
from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS
from autogpt.llm.utils import (
    count_message_tokens,
    count_string_tokens,
    create_chat_completion,
)
from autogpt.logs import (
    PROMPT_SUMMARY_FILE_NAME,
    SUMMARY_FILE_NAME,
    LogCycleHandler,
    logger,
)


@dataclass
class MessageHistory(ChatSequence):
    max_summary_tlength: int = 500
    agent: Optional[BaseAgent | Agent] = None
    summary: str = "I was created"
    last_trimmed_index: int = 0

    SUMMARIZATION_PROMPT = '''Your task is to create a concise running summary of actions and information results in the provided text, focusing on key and potentially important information to remember.

You will receive the current summary and your latest actions. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.

Summary So Far:
"""
{summary}
"""

Latest Development:
"""
{new_events}
"""
'''

    def trim_messages(
        self, current_message_chain: list[Message], config: Config
    ) -> tuple[Message, list[Message]]:
        """
        Returns a list of trimmed messages: messages which are in the message history
        but not in current_message_chain.

        Args:
            current_message_chain (list[Message]): The messages currently in the context.
            config (Config): The config to use.

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

        if not new_messages_not_in_chain:
            return self.summary_message(), []

        new_summary_message = self.update_running_summary(
            new_events=new_messages_not_in_chain, config=config
        )

        # Find the index of the last message processed
        last_message = new_messages_not_in_chain[-1]
        self.last_trimmed_index = self.messages.index(last_message)

        return new_summary_message, new_messages_not_in_chain

    def per_cycle(
        self, messages: Optional[list[Message]] = None
    ) -> Iterator[tuple[Message | None, Message, Message]]:
        """
        Yields:
            Message: a message containing user input
            Message: a message from the AI containing a proposed action
            Message: the message containing the result of the AI's proposed action
        """
        messages = messages or self.messages
        for i in range(0, len(messages) - 1):
            ai_message = messages[i]
            if ai_message.type != "ai_response":
                continue
            user_message = (
                messages[i - 1] if i > 0 and messages[i - 1].role == "user" else None
            )
            result_message = messages[i + 1]
            try:
                assert (
                    extract_dict_from_response(ai_message.content) != {}
                ), "AI response is not a valid JSON object"
                assert result_message.type == "action_result"

                yield user_message, ai_message, result_message
            except AssertionError as err:
                logger.debug(
                    f"Invalid item in message history: {err}; Messages: {messages[i-1:i+2]}"
                )

    def summary_message(self) -> Message:
        return Message(
            "system",
            f"This reminds you of these events from your past: \n{self.summary}",
        )

    def update_running_summary(
        self,
        new_events: list[Message],
        config: Config,
        max_summary_length: Optional[int] = None,
    ) -> Message:
        """
        This function takes a list of Message objects and updates the running summary
        to include the events they describe. The updated summary is returned
        in a Message formatted in the 1st person past tense.

        Args:
            new_events: A list of Messages containing the latest events to be added to the summary.

        Returns:
            Message: a Message containing the updated running summary.

        Example:
            ```py
            new_events = [{"event": "entered the kitchen."}, {"event": "found a scrawled note with the number 7"}]
            update_running_summary(new_events)
            # Returns: "This reminds you of these events from your past: \nI entered the kitchen and found a scrawled note saying 7."
            ```
        """
        if not new_events:
            return self.summary_message()
        if not max_summary_length:
            max_summary_length = self.max_summary_tlength

        # Create a copy of the new_events list to prevent modifying the original list
        new_events = copy.deepcopy(new_events)

        # Replace "assistant" with "you". This produces much better first person past tense results.
        for event in new_events:
            if event.role.lower() == "assistant":
                event.role = "you"

                # Remove "thoughts" dictionary from "content"
                try:
                    content_dict = extract_dict_from_response(event.content)
                    if "thoughts" in content_dict:
                        del content_dict["thoughts"]
                    event.content = json.dumps(content_dict)
                except json.JSONDecodeError as e:
                    logger.error(f"Error: Invalid JSON: {e}")
                    if config.debug_mode:
                        logger.error(f"{event.content}")

            elif event.role.lower() == "system":
                event.role = "your computer"

            # Delete all user messages
            elif event.role == "user":
                new_events.remove(event)

        summ_model = OPEN_AI_CHAT_MODELS[config.fast_llm]

        # Determine token lengths for use in batching
        prompt_template_length = len(
            MessageHistory.SUMMARIZATION_PROMPT.format(summary="", new_events="")
        )
        max_input_tokens = summ_model.max_tokens - max_summary_length
        summary_tlength = count_string_tokens(self.summary, summ_model.name)
        batch: list[Message] = []
        batch_tlength = 0

        # TODO: Put a cap on length of total new events and drop some previous events to
        # save API cost. Need to think thru more how to do it without losing the context.
        for event in new_events:
            event_tlength = count_message_tokens(event, summ_model.name)

            if (
                batch_tlength + event_tlength
                > max_input_tokens - prompt_template_length - summary_tlength
            ):
                # The batch is full. Summarize it and start a new one.
                self._update_summary_with_batch(batch, config, max_summary_length)
                summary_tlength = count_string_tokens(self.summary, summ_model.name)
                batch = [event]
                batch_tlength = event_tlength
            else:
                batch.append(event)
                batch_tlength += event_tlength

        if batch:
            # There's an unprocessed batch. Summarize it.
            self._update_summary_with_batch(batch, config, max_summary_length)

        return self.summary_message()

    def _update_summary_with_batch(
        self, new_events_batch: list[Message], config: Config, max_output_length: int
    ) -> None:
        prompt = MessageHistory.SUMMARIZATION_PROMPT.format(
            summary=self.summary, new_events=new_events_batch
        )

        prompt = ChatSequence.for_model(config.fast_llm, [Message("user", prompt)])
        if (
            self.agent is not None
            and hasattr(self.agent, "created_at")
            and isinstance(
                getattr(self.agent, "log_cycle_handler", None), LogCycleHandler
            )
        ):
            self.agent.log_cycle_handler.log_cycle(
                self.agent.ai_config.ai_name,
                self.agent.created_at,
                self.agent.cycle_count,
                prompt.raw(),
                PROMPT_SUMMARY_FILE_NAME,
            )

        self.summary = create_chat_completion(
            prompt, config, max_tokens=max_output_length
        ).content

        if (
            self.agent is not None
            and hasattr(self.agent, "created_at")
            and isinstance(
                getattr(self.agent, "log_cycle_handler", None), LogCycleHandler
            )
        ):
            self.agent.log_cycle_handler.log_cycle(
                self.agent.ai_config.ai_name,
                self.agent.created_at,
                self.agent.cycle_count,
                self.summary,
                SUMMARY_FILE_NAME,
            )
