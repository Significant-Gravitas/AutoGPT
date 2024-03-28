import logging
from typing import TYPE_CHECKING

from autogpt.agents.prompt_strategies.one_shot import OneShotAgentPromptStrategy
from autogpt.agents.protocols import BuildPrompt, ParseResponse, Single
from autogpt.agents.components import Single

from autogpt.config.config import Config
from autogpt.core.prompting.schema import ChatPrompt
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatMessage,
    ChatModelInfo,
)
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.models.command import Command
from autogpt.agents.base import ThoughtProcessOutput
from autogpt.agents.components import Component

if TYPE_CHECKING:
    from autogpt.agents.agent import AgentSettings


logger = logging.getLogger(__name__)


class OneShotComponent(Component, BuildPrompt, ParseResponse):
    def __init__(
        self,
        settings: "AgentSettings",
        legacy_config: Config,
        llm_provider,
        send_token_limit: int,
        llm: ChatModelInfo,
    ):
        # TODO kcze temp
        self.settings = settings
        self.legacy_config = legacy_config
        self.llm_provider = llm_provider
        self.send_token_limit = send_token_limit
        self.llm = llm
        self.prompt_strategy = OneShotAgentPromptStrategy(
            configuration=settings.prompt_config,
            logger=logger,
        )

    def build_prompt(
        self, messages: list[ChatMessage], commands: list[Command], prompt: ChatPrompt
    ) -> Single[ChatPrompt]:
        ai_directives = self.settings.directives.copy(deep=True)
        # ai_directives.resources += scratchpad.resources
        # ai_directives.constraints += scratchpad.constraints
        # ai_directives.best_practices += scratchpad.best_practices
        # extra_commands += list(scratchpad.commands.values())

        # if include_os_info is None:
        include_os_info = self.legacy_config.execute_local_commands

        # Override the prompt with the one-shot prompt
        return Single(self.prompt_strategy.build_prompt(
            task=self.settings.task,
            ai_profile=self.settings.ai_profile,
            ai_directives=ai_directives,
            commands=get_openai_command_specs(commands),
            event_history=self.settings.history.episodes,
            max_prompt_tokens=self.send_token_limit,
            count_tokens=lambda x: self.llm_provider.count_tokens(x, self.llm.name),
            count_message_tokens=lambda x: self.llm_provider.count_message_tokens(
                x, self.llm.name
            ),
            extra_messages=messages,
            include_os_info=include_os_info,
        ))

    def parse_response(
        self, result: ThoughtProcessOutput, llm_response: AssistantChatMessage
    ) -> Single[ThoughtProcessOutput]:
        (
            command_name,
            arguments,
            assistant_reply_dict,
        ) = self.prompt_strategy.parse_response_content(llm_response)

        # TODO kcze overwrite pipeline result for now
        result.command_name = command_name
        result.command_args = arguments
        result.thoughts = assistant_reply_dict
        return Single(result)
