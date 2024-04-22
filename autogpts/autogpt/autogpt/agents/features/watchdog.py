import logging

from autogpt.agents.base import ThoughtProcessOutput
from autogpt.agents.components import ComponentSystemError
from autogpt.agents.features.context import ContextComponent
from autogpt.agents.protocols import AfterParse
from autogpt.models.action_history import EpisodicActionHistory

from ..base import BaseAgentConfiguration

logger = logging.getLogger(__name__)


class WatchdogComponent(AfterParse):
    """
    Adds a watchdog feature to an agent class. Whenever the agent starts
    looping, the watchdog will switch from the FAST_LLM to the SMART_LLM and re-think.
    """

    run_after = [ContextComponent]

    def __init__(
        self, config: BaseAgentConfiguration, event_history: EpisodicActionHistory
    ):
        self.config = config
        self.event_history = event_history
        self.revert_big_brain = False

    def after_parse(self, result: ThoughtProcessOutput) -> None:
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
                # Trigger retry of all pipelines prior to this component
                raise ComponentSystemError()
