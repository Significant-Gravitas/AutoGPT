from logging import Logger
import uuid
import enum
from typing import Optional
from pydantic import BaseModel

from autogpt.core.configuration import SystemConfiguration, UserConfigurable

from autogpt.core.utils.json_schema import JSONSchema

from autogpt.core.prompting.utils.utils import json_loads, to_numbered_list, to_string_list
from autogpt.core.prompting.base import (
    BasePromptStrategy,
    PromptStrategiesConfiguration,
)
from autogpt.core.prompting.schema import (
    LanguageModelClassification,
)

from autogpt.core.resource.model_providers import (
    CompletionModelFunction,
    ChatMessage,
    AssistantChatMessageDict,
    ChatPrompt,
)

from autogpt.core.agents.simple.lib.schema import (
    Task,
    TaskType,
)


class InitialPlanFunctionNames(str, enum.Enum):
    INITIAL_PLAN: str = "make_initial_plan"


class InitialPlanStrategyConfiguration(PromptStrategiesConfiguration):
    model_classification: LanguageModelClassification = (
        LanguageModelClassification.SMART_MODEL_8K
    )
    default_function_call: InitialPlanFunctionNames = (
        InitialPlanFunctionNames.INITIAL_PLAN
    )
    strategy_name: str = "make_initial_plan"


class InitialPlanStrategy(BasePromptStrategy):
    default_configuration = InitialPlanStrategyConfiguration()
    STRATEGY_NAME = "make_initial_plan"

    ###
    ### PROMPTS
    ###

    FIRST_SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert project planner. You're responsibility is to create work plans for autonomous agents. "
        "You will be given a name, a role, set of goals for the agent to accomplish. Your job is to "
        "break down those goals into a set of tasks that the agent can accomplish to achieve those goals. "
        "Agents are resourceful, but require clear instructions. Each task you create should have clearly defined "
        "`ready_criteria` that the agent can check to see if the task is ready to be started. Each task should "
        "also have clearly defined `acceptance_criteria` that the agent can check to evaluate if the task is complete. "
        "You should create as many tasks as you think is necessary to accomplish the goals.\n\n"
        "System Info:\n{system_info}"
    )

    DEFAULT_SYSTEM_INFO = [
        "The OS you are running on is: {os_info}",
        #"It takes money to let you run. Your API budget is ${api_budget:.3f}",
        #"The current time and date is {current_time}",
    ]

    DEFAULT_USER_PROMPT_TEMPLATE = (
        "You are {agent_name}\n" "Your goals are:\n" "{agent_goals}"
    )

    ###
    ### FUNCTIONS
    ###

    DEFAULT_CREATE_PLAN_FUNCTION = CompletionModelFunction(
        name=InitialPlanFunctionNames.INITIAL_PLAN.value,
        description="Creates a set of tasks that forms the initial plan for an autonomous agent.",
        parameters={
            "task_list": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "objective": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="An imperative verb phrase that succinctly describes the task.",
                        ),
                        "type": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="A categorization for the task.",
                            enum=[t.value for t in TaskType],
                        ),
                        "acceptance_criteria": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description="A list of measurable and testable criteria that must be met for the task to be considered complete.",
                            ),
                        ),
                        "priority": JSONSchema(
                            type=JSONSchema.Type.INTEGER,
                            description="A number between 1 and 10 indicating the priority of the task relative to other generated tasks.",
                            minimum=1,
                            maximum=10,
                        ),
                        "ready_criteria": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description="A list of measurable and testable criteria that must be met before the task can be started.",
                            ),
                        ),
                    },
                ),
            ),
        },
    )

    def __init__(
        self,
        logger: Logger,
        model_classification: LanguageModelClassification,
        default_function_call: InitialPlanFunctionNames,
        strategy_name: str,
    ):
        self._logger = logger
        self._model_classification = model_classification

        self._system_prompt_template = self.FIRST_SYSTEM_PROMPT_TEMPLATE
        self._system_info = self.DEFAULT_SYSTEM_INFO
        self._user_prompt_template = self.DEFAULT_USER_PROMPT_TEMPLATE
        self._strategy_functions = [self.DEFAULT_CREATE_PLAN_FUNCTION]

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        agent_goal_sentence: str,
        tools: list[str],
        os_info: str,
        api_budget: float,
        current_time: str,
        **kwargs,
    ) -> ChatPrompt:
        template_kwargs = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "os_info": os_info,
            "api_budget": api_budget,
            "current_time": current_time,
            **kwargs,
        }
        template_kwargs["agent_goals"] = (
            to_numbered_list(agent_goals, **template_kwargs),
        )
        template_kwargs["agent_goal_sentence"] = (agent_goal_sentence,)
        template_kwargs["tools"] = to_numbered_list(tools, **template_kwargs)
        template_kwargs["system_info"] = to_numbered_list(
            self._system_info, **template_kwargs
        )

        system_prompt = ChatMessage.system(
            content=self._system_prompt_template.format(**template_kwargs),
        )
        user_prompt = ChatMessage.user(
            content=self._user_prompt_template.format(**template_kwargs),
        )
        strategy_functions = self._strategy_functions

        return ChatPrompt(
            messages=[system_prompt, user_prompt],
            functions=strategy_functions,
            function_call=InitialPlanFunctionNames.INITIAL_PLAN.value,
            default_function_call=InitialPlanFunctionNames.INITIAL_PLAN.value,
            # TODO:
            tokens_used=0,
        )

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        parsed_response = json_loads(response_content["function_call"]["arguments"])
        parsed_response["task_list"] = [
            Task.parse_obj(task) for task in parsed_response["task_list"]
        ]
        return parsed_response
