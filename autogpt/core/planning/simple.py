import logging

from autogpt.core.configuration import (
    SystemSettings,
    SystemConfiguration,
    Configurable,
)
from autogpt.core.planning.base import (
    ModelMessage,
    ModelPrompt,
    ModelRole,
    Planner,
    PlanningPromptContext,
    SelfFeedbackPromptContext,
)
from autogpt.core.workspace import Workspace

DEFAULT_USER_OBJECTIVE = (
    "Write a wikipedia style article about the project: "
    "https://github.com/significant-gravitas/Auto-GPT"
)
DEFAULT_TRIGGERING_PROMPT = (
    "Determine which next command to use, and respond using the format specified above:"
)

DEFAULT_OBJECTIVE_SYSTEM_PROMPT = (
    "Your task is to devise up to 5 highly effective goals and an appropriate "
    "role-based name (_GPT) for an autonomous agent, ensuring that the goals are "
    "optimally aligned with the successful completion of its assigned task.\n\n"
    "The user will provide the task, you will provide only the output in the exact "
    "format specified below with no explanation or conversation.\n\n"
    "Example input:\n"
    "Help me with marketing my business\n\n"
    "Example output:\n"
    "Name: CMOGPT\n\n"
    "Description: a professional digital marketer AI that assists Solopreneurs in "
    "growing their businesses by providing world-class expertise in solving "
    "marketing problems for SaaS, content products, agencies, and more.\n\n"
    "Goals:\n"
    "- Engage in effective problem-solving, prioritization, planning, and supporting "
    "execution to address your marketing needs as your virtual Chief Marketing "
    "Officer.\n\n"
    "- Provide specific, actionable, and concise advice to help you make informed "
    "decisions without the use of platitudes or overly wordy explanations.\n\n"
    "- Identify and prioritize quick wins and cost-effective campaigns that maximize "
    "results with minimal time and budget investment.\n\n"
    "- Proactively take the lead in guiding you and offering suggestions when faced "
    "with unclear information or uncertainty to ensure your marketing strategy "
    "remains on track.\n\n"
)

DEFAULT_OBJECTIVE_USER_PROMPT_TEMPLATE = (
    "Task: '{user_objective}'\n"
    "Respond only with the output in the exact format specified in the "
    "system prompt, with no explanation or conversation.\n"
)


class PlannerConfiguration(SystemConfiguration):
    """Configuration for the Planner subsystem."""
    agent_name: str
    agent_role: str
    agent_goals: list[str]


class SimplePlanner(Planner, Configurable):
    """Manages the agent's planning and goal-setting by constructing language model prompts."""

    defaults = SystemSettings(
        name="planner",
        description="Manages the agent's planning and goal-setting by constructing language model prompts.",
        configuration=PlannerConfiguration(
            agent_name="Entrepreneur-GPT",
            agent_role=(
                "An AI designed to autonomously develop and run businesses with "
                "the sole goal of increasing your net worth."
            ),
            agent_goals=[
                "Increase net worth",
                "Grow Twitter Account",
                "Develop and manage multiple businesses autonomously",
            ],
        )
    )

    def __init__(
        self,
        configuration: PlannerConfiguration,
        logger: logging.Logger = None,  # Logger is not available during bootstrapping.
        workspace: Workspace = None,    # Workspace is not available during bootstrapping.
    ) -> None:
        self._configuration = configuration
        self._logger = logger
        self._workspace = workspace

    @staticmethod
    def construct_objective_prompt_from_user_input(user_objective: str) -> ModelPrompt:
        system_message = ModelMessage(
            role=ModelRole.SYSTEM,
            content=DEFAULT_OBJECTIVE_SYSTEM_PROMPT,
        )
        user_message = ModelMessage(
            role=ModelRole.USER,
            content=DEFAULT_OBJECTIVE_USER_PROMPT_TEMPLATE.format(user_objective),
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
