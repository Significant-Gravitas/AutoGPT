from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.planning.base import PromptStrategy
from autogpt.core.planning.schema import (
    LanguageModelClassification,
    LanguageModelPrompt,
)
from autogpt.core.planning.strategies.utils import json_loads
from autogpt.core.resource.model_providers import (
    LanguageModelFunction,
    LanguageModelMessage,
    MessageRole,
)


class NameAndGoalsConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable()
    system_prompt: str = UserConfigurable()
    user_prompt_template: str = UserConfigurable()
    create_agent_function: dict = UserConfigurable()


class NameAndGoals(PromptStrategy):
    DEFAULT_SYSTEM_PROMPT = (
        "Your job is to respond to a user-defined task by invoking the `create_agent` function "
        "to generate an autonomous agent to complete the task. You should supply a role-based "
        "name for the agent, an informative description for what the agent does, and 1 to 5 "
        "goals that are optimally aligned with the successful completion of its assigned task.\n\n"
        "Example Input:\n"
        "Help me with marketing my business\n\n"
        "Example Function Call:\n"
        "create_agent(name='CMOGPT', "
        "description='A professional digital marketer AI that assists Solopreneurs in "
        "growing their businesses by providing world-class expertise in solving "
        "marketing problems for SaaS, content products, agencies, and more.', "
        "goals=['Engage in effective problem-solving, prioritization, planning, and "
        "supporting execution to address your marketing needs as your virtual Chief "
        "Marketing Officer.', 'Provide specific, actionable, and concise advice to "
        "help you make informed decisions without the use of platitudes or overly "
        "wordy explanations.', 'Identify and prioritize quick wins and cost-effective "
        "campaigns that maximize results with minimal time and budget investment.', "
        "'Proactively take the lead in guiding you and offering suggestions when faced "
        "with unclear information or uncertainty to ensure your marketing strategy "
        "remains on track.'])"
    )

    DEFAULT_USER_PROMPT_TEMPLATE = "'{user_objective}'"

    DEFAULT_CREATE_AGENT_FUNCTION = {
        "name": "create_agent",
        "description": ("Create a new autonomous AI agent to complete a given task."),
        "parameters": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "A short role-based name for an autonomous agent.",
                },
                "agent_role": {
                    "type": "string",
                    "description": "An informative one sentence description of what the AI agent does",
                },
                "agent_goals": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "One to five highly effective goals that are optimally aligned with the completion of a "
                        "specific task. The number and complexity of the goals should correspond to the "
                        "complexity of the agent's primary objective."
                    ),
                },
            },
            "required": ["agent_name", "agent_role", "agent_goals"],
        },
    }

    default_configuration = NameAndGoalsConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        create_agent_function=DEFAULT_CREATE_AGENT_FUNCTION,
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt: str,
        user_prompt_template: str,
        create_agent_function: str,
    ):
        self._model_classification = model_classification
        self._system_prompt_message = system_prompt
        self._user_prompt_template = user_prompt_template
        self._create_agent_function = create_agent_function

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(self, user_objective: str = "", **kwargs) -> LanguageModelPrompt:
        system_message = LanguageModelMessage(
            role=MessageRole.SYSTEM,
            content=self._system_prompt_message,
        )
        user_message = LanguageModelMessage(
            role=MessageRole.USER,
            content=self._user_prompt_template.format(
                user_objective=user_objective,
            ),
        )
        create_agent_function = LanguageModelFunction(
            json_schema=self._create_agent_function,
        )
        prompt = LanguageModelPrompt(
            messages=[system_message, user_message],
            functions=[create_agent_function],
            # TODO
            tokens_used=0,
        )
        return prompt

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
        parsed_response = json_loads(response_content["function_call"]["arguments"])
        return parsed_response
