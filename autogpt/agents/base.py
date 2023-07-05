from __future__ import annotations

from abc import ABCMeta, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Optional

from colorama import Fore

from autogpt.config import AIConfig, Config
from autogpt.llm.base import ChatModelResponse, ChatSequence, Message
from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS
from autogpt.llm.utils import count_message_tokens, create_chat_completion
from autogpt.logs import logger
from autogpt.memory.message_history import MessageHistory
from autogpt.models.command_registry import CommandRegistry
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT


class BaseAgent(metaclass=ABCMeta):
    def __init__(
        self,
        ai_config: AIConfig,
        command_registry: CommandRegistry,
        config: Config,
        big_brain=True,
        default_cycle_instruction: str = DEFAULT_TRIGGERING_PROMPT,
        cycle_budget: Optional[int] = None,
        send_token_limit: Optional[int] = None,
        summary_max_tlength: Optional[int] = None,
    ):
        self.ai_config = ai_config
        self.command_registry = command_registry
        self.config = config
        self.big_brain = big_brain
        self.default_cycle_instruction = default_cycle_instruction
        self.cycle_budget = self.cycles_remaining = cycle_budget
        self.cycle_count = 0

        self.system_prompt = ai_config.construct_full_prompt(config)
        if config.debug_mode:
            logger.typewriter_log(
                f"{ai_config.ai_name} System Prompt:", Fore.GREEN, self.system_prompt
            )

        llm_name = (
            self.config.smart_llm_model
            if self.big_brain
            else self.config.fast_llm_model
        )
        self.llm = OPEN_AI_CHAT_MODELS[llm_name]
        self.send_token_limit = send_token_limit or self.llm.max_tokens * 3 // 4

        self.history = MessageHistory(
            self.llm,
            max_summary_tlength=summary_max_tlength or self.send_token_limit // 6,
        )

    def __next__(self) -> None:
        """Runs the agent for one cycle using its `default_cycle_instruction`.

        Raises `StopIteration` if the `cycle_budget` is reached.
        """

        # stop if cycle budget reached
        if self.cycles_remaining is not None and self.cycles_remaining <= 0:
            raise StopIteration

        self.think(self.default_cycle_instruction)

        self.cycle_count += 1
        if self.cycles_remaining is not None:
            self.cycles_remaining -= 1

    def think(self, instruction: str) -> None:
        """Runs the agent for one cycle.

        Params:
            instruction: The instruction to put at the end of the prompt.
        """

        prompt: ChatSequence = self.construct_prompt(instruction)

        self.on_before_think(prompt, instruction)

        with self.context_while_think():
            raw_response = create_chat_completion(prompt, self.config)

        self.on_response(raw_response, prompt, instruction)

    def construct_base_prompt(
        self,
        prepend_messages: list[Message] = [],
        append_messages: list[Message] = [],
        reserve_tokens: int = 0,
    ) -> ChatSequence:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. `prepend_messages`
        3. Message history of the agent, truncated & prepended with running summary as needed
        4. `append_messages`

        Params:
            prepend_messages: Messages to insert between the system prompt and message history
            append_messages: Messages to insert after the message history
            reserve_tokens: Number of tokens to reserve for content that is added later
        """

        prompt = ChatSequence.for_model(
            self.llm.name,
            [Message("system", self.system_prompt)] + prepend_messages,
        )

        # Reserve tokens for messages to be appended later, if any
        reserve_tokens += self.history.max_summary_tlength
        if append_messages:
            reserve_tokens += count_message_tokens(append_messages, self.llm.name)

        # Fill message history, up to a margin of reserved_tokens.
        # Trim remaining historical messages and add them to the running summary.
        history_start_index = len(prompt)
        trimmed_history = add_history_upto_token_limit(
            prompt, self.history, self.send_token_limit - reserve_tokens
        )
        new_summary_msg = self.history.update_running_summary(
            trimmed_history, self.config
        )
        prompt.insert(history_start_index, new_summary_msg)

        if append_messages:
            prompt.extend(append_messages)

        return prompt

    def construct_prompt(self, cycle_instruction: str) -> ChatSequence:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        2. Message history of the agent, truncated & prepended with running summary as needed
        3. `cycle_instruction`

        Params:
            cycle_instruction: The final instruction for a thinking cycle
        """
        cycle_instruction_msg = Message("user", cycle_instruction)
        cycle_instruction_tlength = count_message_tokens(
            cycle_instruction_msg, self.llm.name
        )
        prompt = self.construct_base_prompt(reserve_tokens=cycle_instruction_tlength)

        # ADD user input message ("triggering prompt")
        prompt.append(cycle_instruction_msg)

        return prompt

    def on_before_think(self, prompt: ChatSequence, instruction: str) -> None:
        """Called after constructing the prompt but before executing it.

        Calls the `on_planning` hook of any enabled and capable plugins, adding their
        output to the prompt.

        Params:
            prompt: The prompt that is about to be executed
            instruction: The instruction for the current cycle, also used in constructing the prompt
        """

        current_tokens_used = prompt.token_length
        plugin_count = len(self.config.plugins)
        for i, plugin in enumerate(self.config.plugins):
            if not plugin.can_handle_on_planning():
                continue
            plugin_response = plugin.on_planning(
                self.ai_config.prompt_generator, prompt.raw()
            )
            if not plugin_response or plugin_response == "":
                continue
            message_to_add = Message("system", plugin_response)
            tokens_to_add = count_message_tokens(message_to_add, self.llm.name)
            if current_tokens_used + tokens_to_add > self.send_token_limit:
                logger.debug(f"Plugin response too long, skipping: {plugin_response}")
                logger.debug(f"Plugins remaining at stop: {plugin_count - i}")
                break
            prompt.insert(
                -1, message_to_add
            )  # HACK: assumes cycle instruction to be at the end
            current_tokens_used += tokens_to_add

    def context_while_think(self) -> AbstractContextManager:
        return nullcontext()

    def on_response(
        self, llm_response: ChatModelResponse, prompt: ChatSequence, instruction: str
    ) -> None:
        """Called upon receiving a response from the chat model.

        Adds the last/newest message in the prompt and the response to `history`,
        and calls `self.parse_and_process_response()` to do the rest.

        Params:
            llm_response: The raw response from the chat model
            prompt: The prompt that was executed
            instruction: The instruction for the current cycle, also used in constructing the prompt
        """

        # Save assistant reply to message history
        self.history.append(prompt[-1])
        self.history.add(
            "assistant", llm_response.content, "ai_response"
        )  # FIXME: support function calls

        try:
            self.parse_and_process_response(llm_response, prompt, instruction)
        except SyntaxError as e:
            logger.error(f"Response could not be parsed: {e}")
            # TODO: tune this message
            self.history.add(
                "system",
                f"Your response could not be parsed: {e}"
                "\n\nRemember to only respond using the specified format above!",
            )
            return

        # TODO: update memory/context

    @abstractmethod
    def parse_and_process_response(
        self, llm_response: ChatModelResponse, prompt: ChatSequence, instruction: str
    ):
        """Validate, parse & process the LLM's response.

        Must be implemented by derivative classes: no base implementation is provided,
        since the implementation depends on the role of the derivative Agent.

        Params:
            llm_response: The raw response from the chat model
            prompt: The prompt that was executed
            instruction: The instruction for the current cycle, also used in constructing the prompt
        """
        pass

    def reset_budget(self, new_budget: int | None = None) -> None:
        self.cycle_budget = self.cycles_remaining = new_budget


def add_history_upto_token_limit(
    prompt: ChatSequence, history: MessageHistory, t_limit: int
):
    current_prompt_length = prompt.token_length
    insertion_index = len(prompt)
    limit_reached = False
    trimmed_messages: list[Message] = []
    for cycle in reversed(list(history.per_cycle())):
        messages_to_add = [msg for msg in cycle if msg is not None]
        tokens_to_add = count_message_tokens(messages_to_add, prompt.model.name)
        if current_prompt_length + tokens_to_add > t_limit:
            limit_reached = True

        if not limit_reached:
            # Add the most recent message to the start of the chain,
            #  after the system prompts.
            prompt.insert(insertion_index, *messages_to_add)
            current_prompt_length += tokens_to_add
        else:
            trimmed_messages = messages_to_add + trimmed_messages

    return trimmed_messages
