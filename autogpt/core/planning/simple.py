import logging

from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.planning import templates
from autogpt.core.planning.base import (
    ModelMessage,
    ModelPrompt,
    ModelRole,
    Planner,
    PlanningPromptContext,
    SelfFeedbackPromptContext,
)
from autogpt.core.workspace import Workspace


class PlannerConfiguration(SystemConfiguration):
    """Configuration for the Planner subsystem."""

    agent_name: str
    agent_role: str
    agent_goals: list[str]


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
        ),
    )

    def __init__(
        self,
        settings: PlannerSettings,
        logger: logging.Logger,
        workspace: Workspace = None,  # Workspace is not available during bootstrapping.
    ) -> None:
        self._configuration = settings.configuration
        self._logger = logger
        self._workspace = workspace

    @staticmethod
    def construct_objective_prompt_from_user_input(user_objective: str) -> ModelPrompt:
        system_message = ModelMessage(
            role=ModelRole.SYSTEM,
            content=templates.OBJECTIVE_SYSTEM_PROMPT,
        )
        user_message = ModelMessage(
            role=ModelRole.USER,
            content=templates.DEFAULT_OBJECTIVE_USER_PROMPT_TEMPLATE.format(
                user_objective=user_objective,
            ),
        )
        return [system_message, user_message]

    def construct_planning_prompt_from_context(
        self,
        context: PlanningPromptContext,
    ) -> ModelPrompt:
        raise NotImplementedError

    def get_self_feedback_prompt(
        self,
        context: SelfFeedbackPromptContext,
    ) -> ModelPrompt:
        raise NotImplementedError
