from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from autogpt.config import AIConfig, Config

    from autogpt.models.command_registry import CommandRegistry

from autogpt.llm.base import ChatModelResponse, ChatSequence, Message
from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS, get_openai_command_specs
from autogpt.llm.utils import count_message_tokens, create_chat_completion
from autogpt.logs import logger
from autogpt.memory.message_history import MessageHistory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT

CommandName = str
CommandArgs = dict[str, str]
AgentThoughts = dict[str, Any]


class BaseAgent(metaclass=ABCMeta):
    """Base class for all Auto-GPT agents."""

    def __init__(
        self,
        ai_config: AIConfig,
        command_registry: CommandRegistry,
        config: Config,
        big_brain: bool = True,
        default_cycle_instruction: str = DEFAULT_TRIGGERING_PROMPT,
        cycle_budget: Optional[int] = 1,
        send_token_limit: Optional[int] = None,
        summary_max_tlength: Optional[int] = None,
    ):
        self.ai_config = ai_config
        """The AIConfig or "personality" object associated with this agent."""

        self.command_registry = command_registry
        """The registry containing all commands available to the agent."""

        self.config = config
        """The applicable application configuration."""

        self.big_brain = big_brain
        """
        Whether this agent uses the configured smart LLM (default) to think,
        as opposed to the configured fast LLM.
        """

        self.default_cycle_instruction = default_cycle_instruction
        """The default instruction passed to the AI for a thinking cycle."""

        self.cycle_budget = cycle_budget
        """
        The number of cycles that the agent is allowed to run unsupervised.

        `None` for unlimited continuous execution,
        `1` to require user approval for every step,
        `0` to stop the agent.
        """

        self.cycles_remaining = cycle_budget
        """The number of cycles remaining within the `cycle_budget`."""

        self.cycle_count = 0
        """The number of cycles that the agent has run since its initialization."""

        self.system_prompt = ai_config.construct_full_prompt(config)
        """
        The system prompt sets up the AI's personality and explains its goals,
        available resources, and restrictions.
        """

        llm_name = self.config.smart_llm if self.big_brain else self.config.fast_llm
        self.llm = OPEN_AI_CHAT_MODELS[llm_name]
        """The LLM that the agent uses to think."""

        self.send_token_limit = send_token_limit or self.llm.max_tokens * 3 // 4
        """
        The token limit for prompt construction. Should leave room for the completion;
        defaults to 75% of `llm.max_tokens`.
        """

        self.history = MessageHistory(
            self.llm,
            max_summary_tlength=summary_max_tlength or self.send_token_limit // 6,
        )

    def think(
        self,
        instruction: Optional[str] = None,
    ) -> tuple[CommandName | None, CommandArgs | None, AgentThoughts]:
        """Runs the agent for one cycle.

        Params:
            instruction: The instruction to put at the end of the prompt.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """

        instruction = instruction or self.default_cycle_instruction

        prompt: ChatSequence = self.construct_prompt(instruction)
        prompt = self.on_before_think(prompt, instruction)

        raw_response = create_chat_completion(
            prompt,
            self.config,
            functions=get_openai_command_specs(self.command_registry)
            if self.config.openai_functions
            else None,
        )
        self.cycle_count += 1

        return self.on_response(raw_response, prompt, instruction)

    @abstractmethod
    def execute(
        self,
        command_name: str | None,
        command_args: dict[str, str] | None,
        user_input: str | None,
    ) -> str:
        """Executes the given command, if any, and returns the agent's response.

        Params:
            command_name: The name of the command to execute, if any.
            command_args: The arguments to pass to the command, if any.
            user_input: The user's input, if any.

        Returns:
            The results of the command.
        """
        ...

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
        if trimmed_history:
            new_summary_msg, _ = self.history.trim_messages(list(prompt), self.config)
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

        if not cycle_instruction:
            raise ValueError("No instruction given")

        cycle_instruction_msg = Message("user", cycle_instruction)
        cycle_instruction_tlength = count_message_tokens(
            cycle_instruction_msg, self.llm.name
        )
        prompt = self.construct_base_prompt(reserve_tokens=cycle_instruction_tlength)

        # ADD user input message ("triggering prompt")
        prompt.append(cycle_instruction_msg)

        return prompt

    def on_before_think(self, prompt: ChatSequence, instruction: str) -> ChatSequence:
        """Called after constructing the prompt but before executing it.

        Calls the `on_planning` hook of any enabled and capable plugins, adding their
        output to the prompt.

        Params:
            instruction: The instruction for the current cycle, also used in constructing the prompt

        Returns:
            The prompt to execute
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
        return prompt

    def on_response(
        self, llm_response: ChatModelResponse, prompt: ChatSequence, instruction: str
    ) -> tuple[CommandName | None, CommandArgs | None, AgentThoughts]:
        """Called upon receiving a response from the chat model.

        Adds the last/newest message in the prompt and the response to `history`,
        and calls `self.parse_and_process_response()` to do the rest.

        Params:
            llm_response: The raw response from the chat model
            prompt: The prompt that was executed
            instruction: The instruction for the current cycle, also used in constructing the prompt

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """

        # Save assistant reply to message history
        self.history.append(prompt[-1])
        self.history.add(
            "assistant", llm_response.content, "ai_response"
        )  # FIXME: support function calls

        try:
            return self.parse_and_process_response(llm_response, prompt, instruction)
        except SyntaxError as e:
            logger.error(f"Response could not be parsed: {e}")
            # TODO: tune this message
            self.history.add(
                "system",
                f"Your response could not be parsed: {e}"
                "\n\nRemember to only respond using the specified format above!",
            )
            return None, None, {}

        # TODO: update memory/context

    @abstractmethod
    def parse_and_process_response(
        self, llm_response: ChatModelResponse, prompt: ChatSequence, instruction: str
    ) -> tuple[CommandName | None, CommandArgs | None, AgentThoughts]:
        """Validate, parse & process the LLM's response.

        Must be implemented by derivative classes: no base implementation is provided,
        since the implementation depends on the role of the derivative Agent.

        Params:
            llm_response: The raw response from the chat model
            prompt: The prompt that was executed
            instruction: The instruction for the current cycle, also used in constructing the prompt

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """
        pass


def add_history_upto_token_limit(
    prompt: ChatSequence, history: MessageHistory, t_limit: int
) -> list[Message]:
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
