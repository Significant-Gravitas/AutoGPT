import time
import logging
from typing import Iterator

from forge.agent.protocols import CommandProvider, DirectiveProvider, MessageProvider
from forge.command.decorator import command
from forge.config.ai_profile import AIProfile
from autogpt.config.config import Config
from autogpt.core.resource.model_providers.schema import ChatMessage
from forge.json.schema import JSONSchema
from forge.command.command import Command
from forge.exceptions import AgentFinished
from autogpt.utils.utils import DEFAULT_FINISH_COMMAND

logger = logging.getLogger(__name__)


class SystemComponent(DirectiveProvider, MessageProvider, CommandProvider):
    """Component for system messages and commands."""

    def __init__(self, config: Config, profile: AIProfile):
        self.legacy_config = config
        self.profile = profile

    def get_constraints(self) -> Iterator[str]:
        if self.profile.api_budget > 0.0:
            yield (
                f"It takes money to let you run. "
                f"Your API budget is ${self.profile.api_budget:.3f}"
            )

    def get_messages(self) -> Iterator[ChatMessage]:
        # Clock
        yield ChatMessage.system(f"The current time and date is {time.strftime('%c')}")

    def get_commands(self) -> Iterator[Command]:
        yield self.finish

    @command(
        names=[DEFAULT_FINISH_COMMAND],
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
