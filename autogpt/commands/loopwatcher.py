import collections
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from autogpt.logs import logger
from autogpt.models.command import CommandInstance
from common.common import simple_exception_handling


class CommandStatus(Enum):
    NeverSeen = 0
    FirstTime = 1
    ShouldBeIgnored = 2


@dataclass
class ExecutedCommand:
    number: int
    status: CommandStatus


class LoopWatcher:
    def __init__(self):
        self.executed_dict: Dict[int, ExecutedCommand] = collections.defaultdict(
            lambda: ExecutedCommand(0, CommandStatus.NeverSeen)
        )

    @simple_exception_handling(return_on_exc=True)
    def should_stop_on_command(self, cmd: CommandInstance):
        tostop = False

        var = self.executed_dict[hash(cmd)]
        var.number += 1

        if var.status == CommandStatus.NeverSeen:
            var.status = CommandStatus.FirstTime

        elif var.status == CommandStatus.ShouldBeIgnored:
            logger.debug(f"Command {cmd} should be ignored")
            if (
                cmd.command.max_seen_to_stop is not None
                and var.number > cmd.command.max_seen_to_stop
            ):
                logger.info(
                    f"Command {cmd} has been executed {var.number} times, stopping."
                )
                return True

        else:
            if self.config.loopwatcher_stop_on_command and cmd.command.stop_if_looped:
                return True

        return tostop

    def command_authorized(self, hash):
        self.executed_dict[hash].status = CommandStatus.ShouldBeIgnored
