import json

from autogpt.core.configuration import (
    SystemConfiguration,
    UserConfigurable,
)
from autogpt.core.planning import LanguageModelPrompt, LanguageModelClassification
from autogpt.core.planning.base import PromptStrategy
from autogpt.core.planning.strategies.utils import to_numbered_list

from autogpt.core.resource.model_providers import (
    MessageRole,
    LanguageModelMessage,
)


class InitialPlanConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable()
    agent_preamble: str = UserConfigurable()
    agent_info: list[str] = UserConfigurable()
    task_format: dict[str, str | list[str]] = UserConfigurable()
    triggering_prompt_template: str = UserConfigurable()
    system_prompt_template: str = UserConfigurable()


class InitialPlan(PromptStrategy):

    DEFAULT_AGENT_PREAMBLE = (
        "You are {agent_name}, {agent_role}.\n"
        "Your goals are:\n"
        "{agent_goals}"
    )

    DEFAULT_AGENT_INFO = [
        "The OS you are running on is: {os_info}",
        "It takes money to let you run. Your API budget is ${api_budget:.3f}",
        "The current time and date is {current_time}",
    ]

    DEFAULT_TASK_FORMAT = {
        "objective": "an imperative verb phrase",
        "task_type": (
            "a categorization for the task. Examples: "
            "'research', 'write', 'code', 'design', 'test', 'market', 'sell', 'manage'"
        ),
        "acceptance_criteria": ["a list of testable criteria that must be met for the task to be considered complete"],
        "priority": "a number between 1 and 10 indicating the priority of the task relative to other generated tasks",
        "ready_criteria": ["a list of criteria that must be met before the task can be started"],
    }

    DEFAULT_TRIGGERING_PROMPT_TEMPLATE = (
        "You are an augmented language model. This means that you are being used in a larger system "
        "to extend your functionality. This larger system provides you with a set of abilities you can use.\n\n"
        "Abilities:\n{abilities}\n\n"
        "The system will also manage your long term memory by storing information you collect and retrieving "
        "relevant information for the task you are working on. You should rely on the system for this "
        "and not attempt to manage your own memory.\n\n"
        "You will accomplish your goals by breaking them down into "
        "a series of tasks and then executing those tasks one by one.\n"
        "Your first objective is to break down your goals into a series of small tasks. "
        "You should be able to accomplish each task with 1-3 uses of your abilities. "
        "Smaller, well-defined tasks are highly preferable.\n\n"
        "You should provide back a list of up to five tasks in JSON format. "
        "Each task should have the following structure:\n\n"
        "{task_format}\n\n"
        "What is your initial task list?"
    )

    DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
        "{agent_preamble}\n\n"
        "Info:\n{info}\n\n"
        "{triggering_prompt}\n\n"
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        agent_preamble: str,
        agent_info: list[str],
        task_format: dict[str, str | list[str]],
        triggering_prompt_template: str,
        system_prompt_template: str,
    ):
        self._model_classification = model_classification
        self._agent_preamble = agent_preamble
        self._agent_info = agent_info
        self._task_format = task_format
        self._triggering_prompt_template = triggering_prompt_template
        self._system_prompt_template = system_prompt_template

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self,
        *_,
        **kwargs,
    ) -> LanguageModelPrompt:
        if "abilities" not in kwargs:
            raise ValueError("You must provide a list of abilities to the initial plan.")

        kwargs["task_format"] = json.dumps(
            self._task_format, indent=4,
        )
        kwargs["agent_goals"] = to_numbered_list(kwargs["agent_goals"], **kwargs)
        kwargs["abilities"] = to_numbered_list(kwargs["abilities"], **kwargs)
        main_prompt_kwargs = {
            "agent_preamble": self._agent_preamble.format(**kwargs),
            "info": to_numbered_list(self._agent_info, **kwargs),
            "triggering_prompt": self._triggering_prompt_template.format(**kwargs),
        }
        main_prompt = LanguageModelMessage(
            role=MessageRole.SYSTEM,
            content=self._system_prompt_template.format(**main_prompt_kwargs),
        )
        return LanguageModelPrompt(
            messages=[main_prompt],
            # TODO:
            tokens_used=0,
        )

    def parse_response_content(self, response_text: str) -> dict:
        return {
            "raw": response_text,
            "parsed": fix_json_using_multiple_techniques(response_text),
        }
