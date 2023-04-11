from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from redis.exceptions import RedisError, ResponseError

if TYPE_CHECKING:
    from redis.asyncio.cluster import ClusterNode


class CommandsParser:
    """
    Parses Redis commands to get command keys.

    COMMAND output is used to determine key locations.
    Commands that do not have a predefined key location are flagged with 'movablekeys',
    and these commands' keys are determined by the command 'COMMAND GETKEYS'.

    NOTE: Due to a bug in redis<7.0, this does not work properly
    for EVAL or EVALSHA when the `numkeys` arg is 0.
     - issue: https://github.com/redis/redis/issues/9493
     - fix: https://github.com/redis/redis/pull/9733

    So, don't use this with EVAL or EVALSHA.
    """

    __slots__ = ("commands", "node")

    def __init__(self) -> None:
        self.commands: Dict[str, Union[int, Dict[str, Any]]] = {}

    async def initialize(self, node: Optional["ClusterNode"] = None) -> None:
        if node:
            self.node = node

        commands = await self.node.execute_command("COMMAND")
        for cmd, command in commands.items():
            if "movablekeys" in command["flags"]:
                commands[cmd] = -1
            elif command["first_key_pos"] == 0 and command["last_key_pos"] == 0:
                commands[cmd] = 0
            elif command["first_key_pos"] == 1 and command["last_key_pos"] == 1:
                commands[cmd] = 1
        self.commands = {cmd.upper(): command for cmd, command in commands.items()}

    # As soon as this PR is merged into Redis, we should reimplement
    # our logic to use COMMAND INFO changes to determine the key positions
    # https://github.com/redis/redis/pull/8324
    async def get_keys(self, *args: Any) -> Optional[Tuple[str, ...]]:
        if len(args) < 2:
            # The command has no keys in it
            return None

        try:
            command = self.commands[args[0]]
        except KeyError:
            # try to split the command name and to take only the main command
            # e.g. 'memory' for 'memory usage'
            args = args[0].split() + list(args[1:])
            cmd_name = args[0].upper()
            if cmd_name not in self.commands:
                # We'll try to reinitialize the commands cache, if the engine
                # version has changed, the commands may not be current
                await self.initialize()
                if cmd_name not in self.commands:
                    raise RedisError(
                        f"{cmd_name} command doesn't exist in Redis commands"
                    )

            command = self.commands[cmd_name]

        if command == 1:
            return (args[1],)
        if command == 0:
            return None
        if command == -1:
            return await self._get_moveable_keys(*args)

        last_key_pos = command["last_key_pos"]
        if last_key_pos < 0:
            last_key_pos = len(args) + last_key_pos
        return args[command["first_key_pos"] : last_key_pos + 1 : command["step_count"]]

    async def _get_moveable_keys(self, *args: Any) -> Optional[Tuple[str, ...]]:
        try:
            keys = await self.node.execute_command("COMMAND GETKEYS", *args)
        except ResponseError as e:
            message = e.__str__()
            if (
                "Invalid arguments" in message
                or "The command has no key arguments" in message
            ):
                return None
            else:
                raise e
        return keys
