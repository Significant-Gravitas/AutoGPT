from __future__ import annotations

import logging
from contextlib import ExitStack

from autogpt.models.action_history import EpisodicActionHistory

from ..base import BaseAgent

logger = logging.getLogger(__name__)


class WatchdogMixin:
    """
    Mixin that adds a watchdog feature to an agent class. Whenever the agent starts
    looping, the watchdog will switch from the FAST_LLM to the SMART_LLM and re-think.
    """

    event_history: EpisodicActionHistory

    def __init__(self, **kwargs) -> None:
        # Initialize other bases first, because we need the event_history from BaseAgent
        super(WatchdogMixin, self).__init__(**kwargs)

        if not isinstance(self, BaseAgent):
            raise NotImplementedError(
                f"{__class__.__name__} can only be applied to BaseAgent derivatives"
            )

    def think(self, *args, **kwargs) -> BaseAgent.ThoughtProcessOutput:
        command_name, command_args, thoughts = super(WatchdogMixin, self).think(
            *args, **kwargs
        )

        if (
            not self.big_brain
            and len(self.event_history) > 1
            and self.config.fast_llm != self.config.smart_llm
        ):
            # Detect repetitive commands
            previous_cycle = self.event_history.episodes[self.event_history.cursor - 1]
            if (
                command_name == previous_cycle.action.name
                and command_args == previous_cycle.action.args
            ):
                logger.info(
                    f"Repetitive command detected ({command_name}), re-thinking with SMART_LLM..."
                )
                with ExitStack() as stack:

                    @stack.callback
                    def restore_state() -> None:
                        # Executed after exiting the ExitStack context
                        self.big_brain = False

                    # Remove partial record of current cycle
                    self.event_history.rewind()

                    # Switch to SMART_LLM and re-think
                    self.big_brain = True
                    return self.think(*args, **kwargs)

        return command_name, command_args, thoughts
