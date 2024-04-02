import logging
import time
from typing import Iterator

from autogpt.llm.api_manager import ApiManager
from autogpt.command_decorator import command
from autogpt.agents.utils.exceptions import (
    AgentFinished,
)
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.agents.components import (
    Component,
)
from autogpt.agents.protocols import CommandProvider, MessageProvider
from autogpt.models.command import Command
from autogpt.config.config import Config
from autogpt.core.resource.model_providers.schema import (
    ChatMessage,
)

logger = logging.getLogger(__name__)


class SystemComponent(Component, MessageProvider, CommandProvider):
    """Component for system messages and commands."""
    def __init__(self, config: Config):
        self.legacy_config = config

    def get_messages(self) -> Iterator[ChatMessage]:
        # Clock
        yield ChatMessage.system(f"The current time and date is {time.strftime('%c')}")

        # Add budget information (if any) to prompt
        api_manager = ApiManager()
        if api_manager.get_total_budget() > 0.0:
            remaining_budget = (
                api_manager.get_total_budget() - api_manager.get_total_cost()
            )
            if remaining_budget < 0:
                remaining_budget = 0
            # TODO kcze this is repeated similarly in constraints
            budget_msg = ChatMessage.system(
                f"Your remaining API budget is ${remaining_budget:.3f}"
                + (
                    " BUDGET EXCEEDED! SHUT DOWN!\n\n"
                    if remaining_budget == 0
                    else (
                        " Budget very nearly exceeded! Shut down gracefully!\n\n"
                        if remaining_budget < 0.005
                        else (
                            " Budget nearly exceeded. Finish up.\n\n"
                            if remaining_budget < 0.01
                            else ""
                        )
                    )
                ),
            )
            logger.debug(budget_msg)
            yield budget_msg

    def get_commands(self) -> Iterator[Command]:
        yield Command.from_decorated_function(self.finish)

    @command(
        parameters={
            "reason": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="A summary to the user of how the goals were accomplished",
                required=True,
            ),
        }
    )
    def finish(self, reason: str):
        """Use this to shut down once you have completed your task,
        or when there are insurmountable problems that make it impossible
        for you to finish your task."""
        raise AgentFinished(reason)
