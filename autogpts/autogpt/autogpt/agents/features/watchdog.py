from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import BaseAgentConfiguration

from autogpt.agents.features.context import ContextComponent
from autogpt.models.action_history import EpisodicActionHistory
from autogpt.agents.components import Component, ComponentSystemError, ProposeAction


logger = logging.getLogger(__name__)


class WatchdogComponent(Component, ProposeAction):
    """
    Adds a watchdog feature to an agent class. Whenever the agent starts
    looping, the watchdog will switch from the FAST_LLM to the SMART_LLM and re-think.
    """
    run_after = [ContextComponent]

    def __init__(self, config: BaseAgentConfiguration, event_history: EpisodicActionHistory):
        self.config = config
        self.event_history = event_history
        self.revert_big_brain = False

    def propose_action(self, result: ProposeAction.Result) -> None:
        if self.revert_big_brain:
            self.config.big_brain = False
            self.revert_big_brain = False

        if not self.config.big_brain and self.config.fast_llm != self.config.smart_llm:
            previous_command, previous_command_args = None, None
            if len(self.event_history) > 1:
                # Detect repetitive commands
                previous_cycle = self.event_history.episodes[
                    self.event_history.cursor - 1
                ]
                previous_command = previous_cycle.action.name
                previous_command_args = previous_cycle.action.args

            rethink_reason = ""

            if not result.command_name:
                rethink_reason = "AI did not specify a command"
            elif (
                result.command_name == previous_command
                and result.command_args == previous_command_args
            ):
                rethink_reason = f"Repititive command detected ({result.command_name})"

            if rethink_reason:
                logger.info(f"{rethink_reason}, re-thinking with SMART_LLM...")
                self.event_history.rewind()
                self.big_brain = True
                self.revert_big_brain = True
                # This will trigger retry of all pipelines on the method
                raise ComponentSystemError()
