"""
AutoCorrectionStrategy Module

This module provides strategies and configurations to assist the AI in refining and clarifying user requirements
through an iterative process, based on the COCE Framework.

Classes:
---------
AutoCorrectionFunctionNames: Enum
    Enum class that lists function names used in refining user context.

AutoCorrectionStrategyConfiguration: BaseModel
    Pydantic model that represents the default configurations for the refine user context strategy.

AutoCorrectionStrategy: BasePromptStrategy
    Strategy that guides the AI in refining and clarifying user requirements based on the COCE Framework.

Examples:
---------
To initialize and use the `AutoCorrectionStrategy`:

>>> strategy = AutoCorrectionStrategy(logger, model_classification= PromptStrategyLanguageModelClassification.FAST_MODEL_4K, default_tool_choice=AutoCorrectionFunctionNames.REFINE_REQUIREMENTS, strategy_name="refine_user_context", context_min_tokens=250, context_max_tokens=300)
>>> prompt = strategy.build_prompt(interupt_refinement_process=False, user_objectives="Build a web app")
"""

import enum
import os
import uuid
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AbstractPromptConfiguration,
    AssistantChatMessageDict,
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    DefaultParsedResponse,
    PromptStrategiesConfiguration,
)
from AFAAS.interfaces.prompts.utils.utils import to_md_quotation
from AFAAS.interfaces.task.task import AbstractTask
from AFAAS.lib.message_user_agent import Questions
from AFAAS.lib.utils.json_schema import JSONSchema


class AutoCorrectionStrategyFunctionNames(str, enum.Enum):
    AUTOCORRECTION: str = "afaas_autocorrection"


class AutoCorrectionStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: AutoCorrectionStrategyFunctionNames = (
        AutoCorrectionStrategyFunctionNames.AUTOCORRECTION
    )
    note_to_agent_length: int = 800
    temperature: float = 0.75


class AutoCorrectionStrategy(AbstractPromptStrategy):
    default_configuration = AutoCorrectionStrategyConfiguration()
    STRATEGY_NAME = "base_autocorection"

    ###
    ### PROMPTS
    ###
    def __init__(
        self,
        default_tool_choice: AutoCorrectionStrategyFunctionNames,
        note_to_agent_length: int,
        temperature: float,  # if coding 0.05,
        # top_p: Optional[float] ,
        # max_tokens : Optional[int] ,
        # frequency_penalty: Optional[float], # Avoid repeting oneselfif coding 0.3
        # presence_penalty : Optional[float], # Avoid certain subjects
        count=0,
        user_last_goal="",
        exit_token: str = str(uuid.uuid4()),
        use_message: bool = False,
    ):
        """
        Initialize the AutoCorrectionStrategy.

        Parameters:
        -----------
        logger: Logger
            The logger object.
        model_classification:  PromptStrategyLanguageModelClassification
            Classification of the language model.
        default_tool_choice: AutoCorrectionFunctionNames
            Default function call for the strategy.
        context_min_tokens: int
            Minimum number of tokens in the context.
        context_max_tokens: int
            Maximum number of tokens in the context.
        count: int, optional (default = 0)
            The count for the iterative process.
        user_last_goal: str, optional (default = "")
            Last goal provided by the user.
        exit_token: str, optional
            Token to indicate exit from the process.
        use_message: bool, optional (default = False)
            Flag to determine whether to use messages.
        """
        self._count = count
        self._config = self.default_configuration
        self.note_to_agent_length = note_to_agent_length

    def set_tools(self, **kwargs):
        if "note_to_agent_length" in kwargs:
            self.note_to_agent_length = "note_to_agent_length"

        # FIXME: Will cose issue if multithreading, better return PromptStrategy object in BaseModelResponse
        strategy = self.get_strategy(
            strategy_name=kwargs["corrected_strategy"].STRATEGY_NAME
        )
        self._tools: list[CompletionModelFunction] = strategy.get_tools()

    async def build_message(
        self,
        task: AbstractTask,
        prompt: str,
        corrected_strategy: AbstractPromptStrategy,
        response: str,
        **kwargs,
    ) -> ChatPrompt:
        # Get the directory containing the currently executing script
        current_directory = os.path.dirname(os.path.abspath(__file__))

        file_loader = FileSystemLoader(current_directory)
        env = Environment(loader=file_loader)
        template = env.get_template("autocorrection.jinja")

        autocorrection_param = {
            "step": self.STRATEGY_NAME,
            "task": task,
            "original_prompt": prompt,
            "response": corrected_strategy.get_autocorrection_response(response),
            "to_md_quotation": to_md_quotation,
        }
        content = template.render(autocorrection_param)
        messages = [ChatMessage.system(content)]
        strategy_tools = self.get_tools()

        #
        # Step 5 :
        #
        prompt = ChatPrompt(
            messages=messages,
            tools=strategy_tools,
            tool_choice="auto",
            default_tool_choice=AutoCorrectionStrategyFunctionNames.AUTOCORRECTION,
            # TODO
            tokens_used=0,
        )

        return prompt

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> DefaultParsedResponse:
        return self.default_parse_response_content(response_content)

    def response_format_instruction(self) -> str:
        return super().response_format_instruction()

    def get_llm_provider(self) -> AbstractLanguageModelProvider:
        return super().get_llm_provider()

    def get_prompt_config(self) -> AbstractPromptConfiguration:
        return super().get_prompt_config()
