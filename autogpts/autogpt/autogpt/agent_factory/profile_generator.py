import logging

from autogpt.config import AIDirectives, AIProfile, Config
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.prompting import (
    ChatPrompt,
    LanguageModelClassification,
    PromptStrategy,
)
from autogpt.core.prompting.utils import json_loads
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessageDict,
    ChatMessage,
    ChatModelProvider,
    CompletionModelFunction,
)
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger(__name__)


class AgentProfileGeneratorConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable(
        default=LanguageModelClassification.SMART_MODEL
    )
    system_prompt: str = UserConfigurable(
        default=(
            "Your job is to respond to a user-defined task, given in triple quotes, by "
            "invoking the `create_agent` function to generate an autonomous agent to "
            "complete the task. "
            "You should supply a role-based name for the agent (_GPT), "
            "an informative description for what the agent does, and "
            "1 to 5 directives in each of the categories Best Practices and Constraints, "
            "that are optimally aligned with the successful completion "
            "of its assigned task.\n"
            "\n"
            "Example Input:\n"
            '"""Help me with marketing my business"""\n\n'
            "Example Function Call:\n"
            "```\n"
            "{"
            '"name": "create_agent",'
            ' "arguments": {'
            '"name": "CMOGPT",'
            ' "description": "a professional digital marketer AI that assists Solopreneurs in'
            " growing their businesses by providing world-class expertise in solving"
            ' marketing problems for SaaS, content products, agencies, and more.",'
            ' "directives": {'
            ' "best_practices": ['
            '"Engage in effective problem-solving, prioritization, planning, and'
            " supporting execution to address your marketing needs as your virtual Chief"
            ' Marketing Officer.",'
            ' "Provide specific, actionable, and concise advice to help you make'
            " informed decisions without the use of platitudes or overly wordy"
            ' explanations.",'
            ' "Identify and prioritize quick wins and cost-effective campaigns that'
            ' maximize results with minimal time and budget investment.",'
            ' "Proactively take the lead in guiding you and offering suggestions when'
            " faced with unclear information or uncertainty to ensure your marketing"
            ' strategy remains on track."'
            "],"  # best_practices
            ' "constraints": ['
            '"Do not suggest illegal or unethical plans or strategies.",'
            ' "Take reasonable budgetary limits into account."'
            "]"  # constraints
            "}"  # directives
            "}"  # arguments
            "}\n"
            "```"
        )
    )
    user_prompt_template: str = UserConfigurable(default='"""{user_objective}"""')
    create_agent_function: dict = UserConfigurable(
        default=CompletionModelFunction(
            name="create_agent",
            description="Create a new autonomous AI agent to complete a given task.",
            parameters={
                "name": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="A short role-based name for an autonomous agent.",
                    required=True,
                ),
                "description": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    description="An informative one sentence description of what the AI agent does",
                    required=True,
                ),
                "directives": JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "best_practices": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            minItems=1,
                            maxItems=5,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                            ),
                            description=(
                                "One to five highly effective best practices that are"
                                " optimally aligned with the completion of the given task."
                            ),
                            required=True,
                        ),
                        "constraints": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            minItems=1,
                            maxItems=5,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                            ),
                            description=(
                                "One to five highly effective constraints that are"
                                " optimally aligned with the completion of the given task."
                            ),
                            required=True,
                        ),
                    },
                    required=True,
                ),
            },
        ).schema
    )


class AgentProfileGenerator(PromptStrategy):
    default_configuration: AgentProfileGeneratorConfiguration = (
        AgentProfileGeneratorConfiguration()
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
        )
        return prompt

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> tuple[AIProfile, AIDirectives]:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        try:
            arguments = json_loads(response_content["function_call"]["arguments"])
            ai_profile = AIProfile(
                ai_name=arguments.get("name"),
                ai_role=arguments.get("description"),
            )
            ai_directives = AIDirectives(
                best_practices=arguments["directives"].get("best_practices"),
                constraints=arguments["directives"].get("constraints"),
                resources=[],
            )
        except KeyError:
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        return ai_profile, ai_directives


async def generate_agent_profile_for_task(
    task: str,
    app_config: Config,
    llm_provider: ChatModelProvider,
) -> tuple[AIProfile, AIDirectives]:
    """Generates an AIConfig object from the given string.

    Returns:
    AIConfig: The AIConfig object tailored to the user's input
    """
    agent_profile_generator = AgentProfileGenerator(
        **AgentProfileGenerator.default_configuration.dict()  # HACK
    )

    prompt = agent_profile_generator.build_prompt(task)

    # Call LLM with the string as user input
    output = (
        await llm_provider.create_chat_completion(
            prompt.messages,
            model_name=app_config.smart_llm,
            functions=prompt.functions,
        )
    ).response

    # Debug LLM Output
    logger.debug(f"AI Config Generator Raw Output: {output}")

    # Parse the output
    ai_profile, ai_directives = agent_profile_generator.parse_response_content(output)

    return ai_profile, ai_directives
