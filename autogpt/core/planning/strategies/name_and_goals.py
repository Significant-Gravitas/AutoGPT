import re

from autogpt.core.configuration import (
    SystemConfiguration,
    UserConfigurable,
)
from autogpt.core.planning.schema import LanguageModelPrompt, LanguageModelClassification
from autogpt.core.planning.base import PromptStrategy
from autogpt.core.resource.model_providers import (
    MessageRole,
    LanguageModelMessage,
)


class NameAndGoalsConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable()
    system_prompt: str = UserConfigurable()
    user_prompt_template: str = UserConfigurable()


class NameAndGoals(PromptStrategy):

    DEFAULT_SYSTEM_PROMPT = (
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
        "remains on track."
    )

    DEFAULT_USER_PROMPT_TEMPLATE = (
        "Task: '{user_objective}'\n"
        "Respond only with the output in the exact format specified in the "
        "system prompt, with no explanation or conversation.\n"
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt: str,
        user_prompt_template: str,
    ):
        self._model_classification = model_classification
        self._system_prompt_message = system_prompt
        self._user_prompt_template = user_prompt_template

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
        prompt = LanguageModelPrompt(
            messages=[system_message, user_message],
            # TODO
            tokens_used=0,
        )
        return prompt

    def parse_response_content(
        self,
        response_content: str,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response text from the objective model.

        Returns:
            The parsed response.

        """
        agent_name = re.search(
            r"Name(?:\s*):(?:\s*)(.*)", response_content, re.IGNORECASE
        ).group(1)
        agent_role = (
            re.search(
                r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",
                response_content,
                re.IGNORECASE | re.DOTALL,
            )
            .group(1)
            .strip()
        )
        agent_goals = re.findall(r"(?<=\n)-\s*(.*)", response_content)
        parsed_response = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "agent_goals": agent_goals,
        }
        return parsed_response
