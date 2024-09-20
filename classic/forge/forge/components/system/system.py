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

    def get_resources(self) -> Iterator[str]:
        yield (
            "You are a Large Language Model, trained on millions of pages of text, "
            "including a lot of factual knowledge. Make use of this factual knowledge "
            "to avoid unnecessary gathering of information."
        )

    def get_best_practices(self) -> Iterator[str]:
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

    def get_messages(self) -> Iterator[ChatMessage]:
        # Clock
        yield ChatMessage.system(
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
        },
    )
    def finish(self, reason: str):
        """Use this to shut down once you have completed your task,
        or when there are insurmountable problems that make it impossible
        for you to finish your task."""
        raise AgentFinished(reason)
