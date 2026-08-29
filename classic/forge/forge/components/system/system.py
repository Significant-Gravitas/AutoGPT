import logging
import time
from typing import Iterator

from forge.agent.protocols import CommandProvider, DirectiveProvider, MessageProvider
from forge.command import Command, command
from forge.llm.providers import ChatMessage
from forge.models.json_schema import JSONSchema
from forge.utils.const import FINISH_COMMAND
from forge.utils.exceptions import AgentFinished

logger = logging.getLogger(__name__)


class SystemComponent(DirectiveProvider, MessageProvider, CommandProvider):
    """Component for system messages and commands."""

    def get_constraints(self) -> Iterator[str]:
        yield "Exclusively use the commands listed below."
        yield (
            "You can only act proactively, and are unable to start background jobs or "
            "set up webhooks for yourself. "
            "Take this into account when planning your actions."
        )
        yield (
            "You are unable to interact with physical objects. "
            "If this is absolutely necessary to fulfill a task or objective or "
            "to complete a step, you must ask the user to do it for you. "
            "If the user refuses this, and there is no other way to achieve your "
            "goals, you must terminate to avoid wasting time and energy."
        )
        # Code-specific constraints
        yield (
            "NEVER modify test files to make tests pass. "
            "If tests fail, the bug is in your implementation, not the test."
        )
        yield (
            "NEVER assume a library is available. Check existing imports and "
            "dependency files (package.json, requirements.txt, etc.) first."
        )
        yield "Never expose, log, or commit secrets, API keys, or credentials."

    def get_resources(self) -> Iterator[str]:
        yield (
            "You are a Large Language Model, trained on millions of pages of text, "
            "including a lot of factual knowledge. Make use of this factual knowledge "
            "to avoid unnecessary gathering of information."
        )

    def get_best_practices(self) -> Iterator[str]:
        # General best practices
        yield (
            "Continuously review and analyze your actions to ensure "
            "you are performing to the best of your abilities."
        )
        yield "Constructively self-criticize your big-picture behavior constantly."
        yield "Reflect on past decisions and strategies to refine your approach."
        yield (
            "Every command has a cost, so be smart and efficient. "
            "Aim to complete tasks in the least number of steps."
        )
        yield (
            "Only make use of your information gathering abilities to find "
            "information that you don't yet have knowledge of."
        )
        # Code-specific best practices
        yield (
            "Read files before modifying them. Understand existing code patterns, "
            "conventions, and interfaces first."
        )
        yield (
            "Mimic existing code style, naming conventions, and patterns. "
            "Don't introduce new patterns unless necessary."
        )
        yield (
            "Execute independent operations in parallel when possible "
            "(e.g., reading multiple files at once)."
        )
        yield (
            "After making changes, verify your work - run available "
            "linters, formatters, and tests."
        )
        yield "Fix root causes, not symptoms. Debug systematically."
        yield "Don't add comments unless code is genuinely complex."

    def get_messages(self) -> Iterator[ChatMessage]:
        # Clock
        yield ChatMessage.user(
            f"## Clock\nThe current time and date is {time.strftime('%c')}"
        )

    def get_commands(self) -> Iterator[Command]:
        yield self.finish

    @command(
        names=[FINISH_COMMAND],
        parameters={
            "reason": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="A summary to the user of how the goals were accomplished",
                required=True,
            ),
            "suggested_next_task": JSONSchema(
                type=JSONSchema.Type.STRING,
                description=(
                    "An optional suggested follow-up task based on "
                    "what was accomplished"
                ),
                required=False,
            ),
        },
    )
    def finish(self, reason: str, suggested_next_task: str = ""):
        """Use this to shut down once you have completed your task,
        or when there are insurmountable problems that make it impossible
        for you to finish your task."""
        raise AgentFinished(reason, suggested_next_task=suggested_next_task or None)
