from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.planning.base import PromptStrategy
from autogpt.core.planning.schema import (
    LanguageModelClassification,
    LanguageModelPrompt,
    Task,
)
from autogpt.core.planning.strategies.utils import json_loads, to_numbered_list
from autogpt.core.resource.model_providers import (
    LanguageModelFunction,
    LanguageModelMessage,
    MessageRole,
)


class NextAbilityConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable()
    system_prompt_template: str = UserConfigurable()
    system_info: list[str] = UserConfigurable()
    user_prompt_template: str = UserConfigurable()
    additional_ability_arguments: dict = UserConfigurable()


class NextAbility(PromptStrategy):
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = "System Info:\n{system_info}"

    DEFAULT_SYSTEM_INFO = [
        "The OS you are running on is: {os_info}",
        "It takes money to let you run. Your API budget is ${api_budget:.3f}",
        "The current time and date is {current_time}",
    ]

    DEFAULT_USER_PROMPT_TEMPLATE = (
        "Your current task is is {task_objective}.\n"
        "You have taken {cycle_count} actions on this task already. "
        "Here is the actions you have taken and their results:\n"
        "{action_history}\n\n"
        "Here is additional information that may be useful to you:\n"
        "{additional_info}\n\n"
        "Additionally, you should consider the following:\n"
        "{user_input}\n\n"
        "Your task of {task_objective} is complete when the following acceptance criteria have been met:\n"
        "{acceptance_criteria}\n\n"
        "Please choose one of the provided functions to accomplish this task. "
        "Some tasks may require multiple functions to accomplish. If that is the case, choose the function that "
        "you think is most appropriate for the current situation given your progress so far."
    )

    DEFAULT_ADDITIONAL_ABILITY_ARGUMENTS = {
        "motivation": {
            "type": "string",
            "description": "Your justification for choosing choosing this function instead of a different one.",
        },
        "self_criticism": {
            "type": "string",
            "description": "Thoughtful self-criticism that explains why this function may not be the best choice.",
        },
        "reasoning": {
            "type": "string",
            "description": "Your reasoning for choosing this function taking into account the `motivation` and weighing the `self_criticism`.",
        },
    }

    default_configuration = NextAbilityConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        system_info=DEFAULT_SYSTEM_INFO,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        additional_ability_arguments=DEFAULT_ADDITIONAL_ABILITY_ARGUMENTS,
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt_template: str,
        system_info: list[str],
        user_prompt_template: str,
        additional_ability_arguments: dict,
    ):
        self._model_classification = model_classification
        self._system_prompt_template = system_prompt_template
        self._system_info = system_info
        self._user_prompt_template = user_prompt_template
        self._additional_ability_arguments = additional_ability_arguments

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self,
        task: Task,
        ability_schema: list[dict],
        os_info: str,
        api_budget: float,
        current_time: str,
        **kwargs,
    ) -> LanguageModelPrompt:
        template_kwargs = {
            "os_info": os_info,
            "api_budget": api_budget,
            "current_time": current_time,
            **kwargs,
        }

        for ability in ability_schema:
            ability["parameters"]["properties"].update(
                self._additional_ability_arguments
            )
            ability["parameters"]["required"] += list(
                self._additional_ability_arguments.keys()
            )

        template_kwargs["task_objective"] = task.objective
        template_kwargs["cycle_count"] = task.context.cycle_count
        template_kwargs["action_history"] = to_numbered_list(
            [action.summary() for action in task.context.prior_actions],
            no_items_response="You have not taken any actions yet.",
            **template_kwargs,
        )
        template_kwargs["additional_info"] = to_numbered_list(
            [memory.summary() for memory in task.context.memories]
            + [info for info in task.context.supplementary_info],
            no_items_response="There is no additional information available at this time.",
            **template_kwargs,
        )
        template_kwargs["user_input"] = to_numbered_list(
            [user_input for user_input in task.context.user_input],
            no_items_response="There are no additional considerations at this time.",
            **template_kwargs,
        )
        template_kwargs["acceptance_criteria"] = to_numbered_list(
            [acceptance_criteria for acceptance_criteria in task.acceptance_criteria],
            **template_kwargs,
        )

        template_kwargs["system_info"] = to_numbered_list(
            self._system_info,
            **template_kwargs,
        )

        system_prompt = LanguageModelMessage(
            role=MessageRole.SYSTEM,
            content=self._system_prompt_template.format(**template_kwargs),
        )
        user_prompt = LanguageModelMessage(
            role=MessageRole.USER,
            content=self._user_prompt_template.format(**template_kwargs),
        )
        functions = [
            LanguageModelFunction(json_schema=ability) for ability in ability_schema
        ]

        return LanguageModelPrompt(
            messages=[system_prompt, user_prompt],
            functions=functions,
            # TODO:
            tokens_used=0,
        )

    def parse_response_content(
        self,
        response_content: dict,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        function_name = response_content["function_call"]["name"]
        function_arguments = json_loads(response_content["function_call"]["arguments"])
        parsed_response = {
            "motivation": function_arguments.pop("motivation"),
            "self_criticism": function_arguments.pop("self_criticism"),
            "reasoning": function_arguments.pop("reasoning"),
            "next_ability": function_name,
            "ability_arguments": function_arguments,
        }
        return parsed_response
