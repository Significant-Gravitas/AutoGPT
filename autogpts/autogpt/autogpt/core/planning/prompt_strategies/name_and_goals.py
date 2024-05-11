import logging

from forge.json.model import JSONSchema
from forge.llm.prompting import ChatPrompt, LanguageModelClassification, PromptStrategy
from forge.llm.providers import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
from forge.models.config import SystemConfiguration, UserConfigurable

logger = logging.getLogger(__name__)


class NameAndGoalsConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable()
    system_prompt: str = UserConfigurable()
    user_prompt_template: str = UserConfigurable()
    create_agent_function: dict = UserConfigurable()


class NameAndGoals(PromptStrategy):
    DEFAULT_SYSTEM_PROMPT = (
        "Your job is to respond to a user-defined task, given in triple quotes, by "
        "invoking the `create_agent` function to generate an autonomous agent to "
        "complete the task. "
        "You should supply a role-based name for the agent, "
        "an informative description for what the agent does, and "
        "1 to 5 goals that are optimally aligned with the successful completion of "
        "its assigned task.\n"
        "\n"
        "Example Input:\n"
        '"""Help me with marketing my business"""\n\n'
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

    DEFAULT_USER_PROMPT_TEMPLATE = '"""{user_objective}"""'

    DEFAULT_CREATE_AGENT_FUNCTION = CompletionModelFunction(
        name="create_agent",
        description="Create a new autonomous AI agent to complete a given task.",
        parameters={
            "agent_name": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="A short role-based name for an autonomous agent.",
            ),
            "agent_role": JSONSchema(
                type=JSONSchema.Type.STRING,
                description=(
                    "An informative one sentence description of what the AI agent does"
                ),
            ),
            "agent_goals": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                minItems=1,
                maxItems=5,
                items=JSONSchema(
                    type=JSONSchema.Type.STRING,
                ),
                description=(
                    "One to five highly effective goals that are optimally aligned "
                    "with the completion of a specific task. "
                    "The number and complexity of the goals should correspond to the "
                    "complexity of the agent's primary objective."
                ),
            ),
        },
    )

    default_configuration: NameAndGoalsConfiguration = NameAndGoalsConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        create_agent_function=DEFAULT_CREATE_AGENT_FUNCTION.schema,
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt: str,
        user_prompt_template: str,
        create_agent_function: dict,
    ):
        self._model_classification = model_classification
        self._system_prompt_message = system_prompt
        self._user_prompt_template = user_prompt_template
        self._create_agent_function = CompletionModelFunction.parse(
            create_agent_function
        )

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(self, user_objective: str = "", **kwargs) -> ChatPrompt:
        system_message = ChatMessage.system(self._system_prompt_message)
        user_message = ChatMessage.user(
            self._user_prompt_template.format(
                user_objective=user_objective,
            )
        )
        prompt = ChatPrompt(
            messages=[system_message, user_message],
            functions=[self._create_agent_function],
            # TODO
            tokens_used=0,
        )
        return prompt

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
                    f"LLM did not call {self._create_agent_function} function; "
                    "agent profile creation failed"
                )
            parsed_response = response_content.tool_calls[0].function.arguments
        except KeyError:
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        return parsed_response
