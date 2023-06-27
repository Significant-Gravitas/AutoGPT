from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Callable, Optional
from autogpt.config.ai_config import AIConfig

from autogpt.config.config import Config
from autogpt.llm.base import ChatModelResponse, ChatSequence, Message
from autogpt.llm.utils import create_chat_completion
from autogpt.memory.message_history import MessageHistory
from autogpt.models.command_registry import CommandRegistry


class BaseAgent:
    think_context: Callable[[BaseAgent], AbstractContextManager] = lambda _: nullcontext()

    def __init__(
        self,
        ai_config: AIConfig,
        system_prompt: str,
        command_registry: CommandRegistry,
        config: Config,
        cycle_budget: Optional[int] = None,
    ):
        self.ai_config = ai_config
        self.system_prompt = system_prompt
        self.command_registry = command_registry
        self.config = config
        self.cycle_budget = cycle_budget
        self.cycle_count = 0
        self.history = MessageHistory(self)

    def __next__(self):
        # stop if cycle budget reached
        if self.cycle_budget is not None and self.cycle_count >= self.cycle_budget:
            raise StopIteration

        # pre-think
        prompt: ChatSequence = self.construct_prompt()
        self.on_before_think()

        # think
        with self.think_context(self):
            raw_response = create_chat_completion(prompt, self.config)
            # assistant_reply = chat_with_ai(
            #     self.config,
            #     self,
            #     self.system_prompt,
            #     self.triggering_prompt,
            #     self.fast_token_limit,
            #     self.config.fast_llm_model,
            # )

        # post-think
        self.on_response(raw_response)

        self.cycle_count += 1

    def construct_prompt(self) -> ChatSequence:
        return ChatSequence.for_model(
            self.config.smart_llm_model, [Message("system", self.system_prompt)]
        )

    def on_before_think(self):
        pass

    def on_response(self, response: ChatModelResponse):
        #   parse response
        #   execute commands
        #   update memory
        pass
