from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable, TypeVar

if TYPE_CHECKING:
    from autogpt.models.command import Command

from autogpt.core.resource.model_providers import CompletionModelFunction

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable)


def get_openai_command_specs(
    commands: Iterable[Command],
) -> list[CompletionModelFunction]:
    """Get OpenAI-consumable function specs for the agent's available commands.
    see https://platform.openai.com/docs/guides/gpt/function-calling
    """
    return [
        CompletionModelFunction(
            name=command.names[0],
            description=command.description,
            parameters={param.name: param.spec for param in command.parameters},
        )
        for command in commands
    ]
