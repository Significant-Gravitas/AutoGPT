import logging

from forge.json.model import JSONSchema
from forge.llm.prompting import ChatPrompt, LanguageModelClassification, PromptStrategy
from forge.llm.prompting.utils import to_numbered_list
from forge.llm.providers import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
from forge.models.config import SystemConfiguration, UserConfigurable

from autogpt.core.planning.schema import Task, TaskType

logger = logging.getLogger(__name__)


class InitialPlanConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable()
    system_prompt_template: str = UserConfigurable()
    system_info: list[str] = UserConfigurable()
    user_prompt_template: str = UserConfigurable()
    create_plan_function: dict = UserConfigurable()


class InitialPlan(PromptStrategy):
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert project planner. "
        "Your responsibility is to create work plans for autonomous agents. "
        "You will be given a name, a role, set of goals for the agent to accomplish. "
        "Your job is to break down those goals into a set of tasks that the agent can"
        " accomplish to achieve those goals. "
        "Agents are resourceful, but require clear instructions."
        " Each task you create should have clearly defined `ready_criteria` that the"
        " agent can check to see if the task is ready to be started."
        " Each task should also have clearly defined `acceptance_criteria` that the"
        " agent can check to evaluate if the task is complete. "
        "You should create as many tasks as you think is necessary to accomplish"
        " the goals.\n\n"
        "System Info:\n{system_info}"
    )

    DEFAULT_SYSTEM_INFO = [
        "The OS you are running on is: {os_info}",
        "It takes money to let you run. Your API budget is ${api_budget:.3f}",
        "The current time and date is {current_time}",
    ]

    DEFAULT_USER_PROMPT_TEMPLATE = (
        "You are {agent_name}, {agent_role}\n" "Your goals are:\n" "{agent_goals}"
    )

    DEFAULT_CREATE_PLAN_FUNCTION = CompletionModelFunction(
        name="create_initial_agent_plan",
        description=(
            "Creates a set of tasks that forms the initial plan of an autonomous agent."
        ),
        parameters={
            "task_list": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "objective": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description=(
                                "An imperative verb phrase that succinctly describes "
                                "the task."
                            ),
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
                                description=(
                                    "A list of measurable and testable criteria that "
                                    "must be met for the task to be considered "
                                    "complete."
                                ),
                            ),
                        ),
                        "priority": JSONSchema(
                            type=JSONSchema.Type.INTEGER,
                            description=(
                                "A number between 1 and 10 indicating the priority of "
                                "the task relative to other generated tasks."
                            ),
                            minimum=1,
                            maximum=10,
                        ),
                        "ready_criteria": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description=(
                                    "A list of measurable and testable criteria that "
                                    "must be met before the task can be started."
                                ),
                            ),
                        ),
                    },
                ),
            ),
        },
    )

    default_configuration: InitialPlanConfiguration = InitialPlanConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        system_info=DEFAULT_SYSTEM_INFO,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        create_plan_function=DEFAULT_CREATE_PLAN_FUNCTION.schema,
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt_template: str,
        system_info: list[str],
        user_prompt_template: str,
        create_plan_function: dict,
    ):
        self._model_classification = model_classification
        self._system_prompt_template = system_prompt_template
        self._system_info = system_info
        self._user_prompt_template = user_prompt_template
        self._create_plan_function = CompletionModelFunction.parse(create_plan_function)

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        abilities: list[str],
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
        template_kwargs["agent_goals"] = to_numbered_list(
            agent_goals, **template_kwargs
        )
        template_kwargs["abilities"] = to_numbered_list(abilities, **template_kwargs)
        template_kwargs["system_info"] = to_numbered_list(
            self._system_info, **template_kwargs
        )

        system_prompt = ChatMessage.system(
            self._system_prompt_template.format(**template_kwargs),
        )
        user_prompt = ChatMessage.user(
            self._user_prompt_template.format(**template_kwargs),
        )

        return ChatPrompt(
            messages=[system_prompt, user_prompt],
            functions=[self._create_plan_function],
            # TODO:
            tokens_used=0,
        )

    def parse_response_content(
        self,
        response_content: AssistantChatMessage,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.
        """
        try:
            if not response_content.tool_calls:
                raise ValueError(
                    f"LLM did not call {self._create_plan_function.name} function; "
                    "plan creation failed"
                )
            parsed_response: object = response_content.tool_calls[0].function.arguments
            parsed_response["task_list"] = [
                Task.parse_obj(task) for task in parsed_response["task_list"]
            ]
        except KeyError:
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        return parsed_response
