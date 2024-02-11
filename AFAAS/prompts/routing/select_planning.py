from __future__ import annotations

from langchain_core.messages  import AIMessage , HumanMessage, SystemMessage , ChatMessage

import enum
import os
import uuid
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AbstractPromptConfiguration,
    AssistantChatMessage,
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
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.utils.json_schema import JSONSchema

LOG = AFAASLogger(name=__name__)


class SelectPlanningStrategyFunctionNames(str, enum.Enum):
    MAKE_PLAN: str = "routing_make_plan"


class SelectPlanningStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: SelectPlanningStrategyFunctionNames = (
        SelectPlanningStrategyFunctionNames.MAKE_PLAN
    )
    temperature: float = 0.7


class SelectPlanningStrategy(AbstractPromptStrategy):
    default_configuration = SelectPlanningStrategyConfiguration()
    STRATEGY_NAME = "routing_make_plan"

    def __init__(
        self,
        default_tool_choice: SelectPlanningStrategyFunctionNames,
        temperature: float,  # if coding 0.05
        count=0,
        exit_token: str = str(uuid.uuid4()),
        use_message: bool = False,
    ):
        """
        Initialize the SelectPlanningStrategy.

        Parameters:
        -----------
        logger: Logger
            The logger object.
        model_classification:  PromptStrategyLanguageModelClassification
            Classification of the language model.
        default_tool_choice: SelectPlanningFunctionNames
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

    async def build_message(self, task: AbstractTask, **kwargs) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + await task.debug_dump_str())

        # Get the directory containing the currently executing script
        current_directory = os.path.dirname(os.path.abspath(__file__))

        file_loader = FileSystemLoader(current_directory)
        env = Environment(loader=file_loader)
        template = env.get_template("30_select_planing_logic.jinja")

        planning_options_param = {
            "step": "SELECT_PLANNING",
            "task": task,
            "to_md_quotation": to_md_quotation,
        }

        content = template.render(planning_options_param)
        messages = [SystemMessage(content)]

        #
        # Step 5 :
        #
        prompt = ChatPrompt(
            messages=messages,
            tools=self.get_tools(),
            tool_choice="auto",
            default_tool_choice=SelectPlanningStrategyFunctionNames.MAKE_PLAN,
            # TODO
            tokens_used=0,
        )

        return prompt

    def set_tools(self, **kwargs):
        if "note_to_agent_length" in kwargs:
            self.note_to_agent_length = "note_to_agent_length"

        self.function_make_plan: CompletionModelFunction = CompletionModelFunction(
            name=SelectPlanningStrategyFunctionNames.MAKE_PLAN.value,
            description="Creates a set of tasks that forms the plan for an autonomous agent.",
            parameters={
                "task_list": JSONSchema(
                    type=JSONSchema.Type.ARRAY,
                    items=JSONSchema(
                        type=JSONSchema.Type.OBJECT,
                        properties={
                            "task_id": JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description="UUID of the task",
                                required=True,
                            ),
                            "task_goal": JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description="The main goal or purpose of the task (20 to 30 words).",
                                required=True,
                            ),
                            "long_description": JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description="A very detailed description of the task for (50 words minimum).",
                                required=True,
                            ),
                            "acceptance_criteria": JSONSchema(
                                type=JSONSchema.Type.ARRAY,
                                items=JSONSchema(
                                    type=JSONSchema.Type.STRING,
                                    description="A list of criteria that must be met for the task to be considered complete.",
                                ),
                                minItems=1,
                            ),
                            "predecessors": JSONSchema(
                                type=JSONSchema.Type.ARRAY,
                                items=JSONSchema(
                                    type=JSONSchema.Type.STRING,
                                    description="List of preceeding task identified by their task_id (UUID).",
                                ),
                                minItems=1,
                            ),
                        },
                    ),
                    required=True,
                ),
            },
        )

        self._tools = [
            self.function_make_plan,
        ]

    def parse_response_content(
        self,
        response_content: AssistantChatMessage,
    ) -> DefaultParsedResponse:
        return self.default_parse_response_content(response_content)

    def response_format_instruction(self) -> str:
        return super().response_format_instruction()

    def get_llm_provider(self) -> AbstractLanguageModelProvider:
        return super().get_llm_provider()

    def get_prompt_config(self) -> AbstractPromptConfiguration:
        return super().get_prompt_config()
