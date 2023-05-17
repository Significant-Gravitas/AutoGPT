import logging
import re
from typing import Callable

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.resource.model_providers import (
    ModelProviderName,
    OpenAIModelName,
    LanguageModelProvider,
    LanguageModelMessage,
    MessageRole,
)
from autogpt.core.planning import templates
from autogpt.core.planning.base import Planner
from autogpt.core.planning.schema import (
    LanguageModelClassification,
    LanguageModelPrompt,
    LanguageModelResponse,
    PlanningContext,
    ReflectionContext,
)
from autogpt.core.workspace import Workspace


class LanguageModelConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    model_name: str = UserConfigurable()
    provider_name: ModelProviderName = UserConfigurable()
    temperature: float = UserConfigurable()


class PlannerConfiguration(SystemConfiguration):
    """Configuration for the Planner subsystem."""

    agent_name: str
    agent_role: str
    agent_goals: list[str]
    models: dict[LanguageModelClassification, LanguageModelConfiguration]


class PlannerSettings(SystemSettings):
    """Settings for the Planner subsystem."""

    configuration: PlannerConfiguration


class SimplePlanner(Planner, Configurable):
    """Manages the agent's planning and goal-setting by constructing language model prompts."""

    defaults = PlannerSettings(
        name="planner",
        description="Manages the agent's planning and goal-setting by constructing language model prompts.",
        configuration=PlannerConfiguration(
            agent_name=templates.AGENT_NAME,
            agent_role=templates.AGENT_ROLE,
            agent_goals=templates.AGENT_GOALS,
            models={
                LanguageModelClassification.FAST_MODEL: LanguageModelConfiguration(
                    model_name=OpenAIModelName.GPT3,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
                LanguageModelClassification.SMART_MODEL: LanguageModelConfiguration(
                    model_name=OpenAIModelName.GPT4,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
            },
        ),
    )

    def __init__(
        self,
        settings: PlannerSettings,
        logger: logging.Logger,
        model_providers: dict[ModelProviderName, LanguageModelProvider],
        workspace: Workspace = None,  # Workspace is not available during bootstrapping.
    ) -> None:
        self._configuration = settings.configuration
        self._logger = logger
        self._workspace = workspace

        self._providers: dict[LanguageModelClassification, LanguageModelProvider] = {}
        for model, model_config in self._configuration.models.items():
            self._providers[model] = model_providers[model_config.provider_name]

    async def decide_name_and_goals(self, user_objective: str) -> LanguageModelResponse:
        model_classification = LanguageModelClassification.FAST_MODEL
        parser = self._parse_name_and_goals
        system_message = LanguageModelMessage(
            role=MessageRole.SYSTEM,
            content=templates.OBJECTIVE_SYSTEM_PROMPT,
        )
        user_message = LanguageModelMessage(
            role=MessageRole.USER,
            content=templates.DEFAULT_OBJECTIVE_USER_PROMPT_TEMPLATE.format(
                user_objective=user_objective,
            ),
        )
        prompt = LanguageModelPrompt(
            messages=[system_message, user_message],
            # TODO
            tokens_used=0,
        )
        return await self.chat_with_model(model_classification, prompt, parser)

    async def plan(self, context: PlanningContext) -> LanguageModelResponse:
        model_classification = LanguageModelClassification.SMART_MODEL
        parser = self._parse_agent_action_model_response
        prompt = LanguageModelPrompt(
            messages=[],
            # TODO
            tokens_used=0,
        )
        return await self.chat_with_model(model_classification, prompt, parser)

    async def reflect(self, context: ReflectionContext) -> LanguageModelResponse:
        model_classification = LanguageModelClassification.SMART_MODEL
        parser = self._parse_agent_feedback_model_response
        prompt = LanguageModelPrompt(
            messages=[],
            # TODO
            tokens_used=0,
        )
        return await self.chat_with_model(model_classification, prompt, parser)

    async def chat_with_model(
        self,
        model_classification: LanguageModelClassification,
        prompt: LanguageModelPrompt,
        completion_parser: Callable[[str], dict]
    ) -> LanguageModelResponse:
        model_configuration = self._configuration.models[model_classification].dict()
        del model_configuration["provider_name"]
        provider = self._providers[model_classification]
        response = await provider.create_language_completion(
            model_prompt=prompt.messages,
            **model_configuration,
            completion_parser=completion_parser,
        )
        return LanguageModelResponse.parse_obj(response.dict())

    @staticmethod
    def _parse_name_and_goals(
        response_text: str,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_text: The raw response text from the objective model.

        Returns:
            The parsed response.

        """
        agent_name = re.search(
            r"Name(?:\s*):(?:\s*)(.*)", response_text, re.IGNORECASE
        ).group(1)
        agent_role = (
            re.search(
                r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",
                response_text,
                re.IGNORECASE | re.DOTALL,
            )
            .group(1)
            .strip()
        )
        agent_goals = re.findall(r"(?<=\n)-\s*(.*)", response_text)
        parsed_response = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "agent_goals": agent_goals,
        }
        return parsed_response

    @staticmethod
    def _parse_agent_action_model_response(
        response_text: str,
    ) -> dict:
        pass

    @staticmethod
    def _parse_agent_feedback_model_response(
        response_text: str,
    ) -> dict:
        pass

    #
    # @staticmethod
    # def construct_objective_prompt_from_user_input(user_objective: str) -> ModelPrompt:
    #
    #     return [system_message, user_message]
    #
    # def construct_planning_prompt_from_context(
    #     self,
    #     context: PlanningPromptContext,
    # ) -> ModelPrompt:
    #     raise NotImplementedError
    #
    # def get_self_feedback_prompt(
    #     self,
    #     context: SelfFeedbackPromptContext,
    # ) -> ModelPrompt:
    #     raise NotImplementedError
