import collections
from enum import Enum
from typing import Dict, NamedTuple

from autogpt.logs import logger
from autogpt.models.command import CommandInstance


class CommandStatus(Enum):
    NeverSeen = 0
    FirstTime = 1
    ShouldBeIgnored = 2


class ExecutedCommand(NamedTuple):
    number: int
    status: CommandStatus


class LoopWatcher:
    def __init__(self):
        self.executed_dict: Dict[int, ExecutedCommand] = collections.defaultdict(
            lambda: ExecutedCommand(0, CommandStatus.NeverSeen)
        )

    def should_stop_on_command(self, cmd: CommandInstance):
        try:
            tostop = False
            from frozendict import frozendict

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
                if (
                    self.config.loopwatcher_stop_on_command
                    and cmd.command.stop_if_looped
                ):
                    return True
                    # logger.info(f"Command {cmd} has been executed {var.number} times, stopping.")
                # logger.info("Already executed this shit, stopping.")

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Exception {e} in adding to dic\n")
            tostop = False
        return tostop

    def command_authorized(self, hash):
        self.executed_dict[hash].status = CommandStatus.ShouldBeIgnored
