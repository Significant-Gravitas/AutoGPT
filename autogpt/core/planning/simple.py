import json
import logging
import platform
import re
import time
from typing import Callable

import distro

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.memory import Memory
from autogpt.core.planning import templates

# from autogpt.core.planning.base import Planner
from autogpt.core.planning.schema import (
    LanguageModelClassification,
    LanguageModelPrompt,
    LanguageModelResponse,
    PlanningContext,
    ReflectionContext,
)
from autogpt.core.resource.model_providers import (
    LanguageModelProvider,
    MessageRole,
    ModelMessage,
    ModelProviderName,
    OpenAIModelName,
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


class SimplePlanner(Configurable):
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
        prompt = self._build_name_and_goals_prompt(user_objective)
        parser = self._parse_name_and_goals

        return await self.chat_with_model(model_classification, prompt, parser)

    async def plan(self, user_feedback: str, memory: Memory) -> LanguageModelResponse:
        model_classification = LanguageModelClassification.SMART_MODEL
        model_name = self._configuration.models[model_classification].model_name
        provider = self._providers[model_classification]
        token_limit = provider.get_token_limit(model_name)
        completion_token_min_length = 1000
        send_token_limit = token_limit - completion_token_min_length
        remaining_budget = provider.get_remaining_budget()

        parser = self._parse_agent_action_model_response
        prompt = self._build_plan_prompt(
            user_feedback, memory, send_token_limit, remaining_budget
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
        completion_parser: Callable[[str], dict],
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
    def _build_name_and_goals_prompt(user_objective: str) -> LanguageModelPrompt:
        system_message = ModelMessage(
            role=MessageRole.SYSTEM,
            content=templates.OBJECTIVE_SYSTEM_PROMPT,
        )
        user_message = ModelMessage(
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
        return prompt

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

    def _build_plan_prompt(
        self,
        user_feedback: str,
        memory: Memory,
        send_token_limit: int,
        remaining_budget: int,
    ) -> LanguageModelPrompt:
        template_args = {
            "agent_name": self._configuration.agent_name,
            "agent_role": self._configuration.agent_role,
            "os_info": get_os_info(),
            "api_budget": remaining_budget,
            "current_time": time.strftime("%c"),
            "response_json_structure": json.dumps(
                templates.PLAN_PROMPT_RESPONSE_DICT, indent=4
            ),
        }

        main_prompt_args = {
            "header": templates.PLAN_PROMPT_HEADER.format(**template_args),
            "goals": to_numbered_list(self._configuration.agent_goals, **template_args),
            "info": to_numbered_list(templates.PLAN_PROMPT_INFO, **template_args),
            "constraints": to_numbered_list(
                templates.PLAN_PROMPT_CONSTRAINTS, **template_args
            ),
            "commands": "",  # TODO
            "resources": to_numbered_list(
                templates.PLAN_PROMPT_RESOURCES, **template_args
            ),
            "performance_evaluations": to_numbered_list(
                templates.PLAN_PROMPT_PERFORMANCE_EVALUATIONS, **template_args
            ),
        }
        main_prompt = ModelMessage(
            role=MessageRole.SYSTEM,
            content=templates.PLAN_PROMPT_MAIN.format(**main_prompt_args),
        )
        user_message = ModelMessage(
            role=MessageRole.USER,
            content=user_feedback,
        )

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


def to_numbered_list(items: list[str], **template_args) -> str:
    return "\n".join(
        f"{i+1}. {item.format(**template_args)}" for i, item in enumerate(items)
    )


def get_os_info() -> str:
    os_name = platform.system()
    os_info = (
        platform.platform(terse=True)
        if os_name != "Linux"
        else distro.name(pretty=True)
    )
    return os_info
