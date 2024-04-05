# 🛠️ Commands

Commands a way for the agent to do anything; e.g. intercting with user or APIs and using tools. They are provided by components that implement the `CommandProvider` protocol.

```py
class CommandProvider(Protocol):
    def get_commands(self) -> Iterator[Command]:
        ...
```

## Command decorator

The easiest way to provide a command is to use `command` decorator on a component method and then yield `Command.from_decorated_function(...)`. Each command needs a name, description and a parameter schema using `JSONSchema`. By default method name is used as a command name, and first part of docstring for the description (before `Args:` or `Returns:`) and schema can be provided in the decorator.

- Simplified
- Full

## Direct construction




```py
from autogpt.agents.components import Component
from autogpt.agents.protocols import CommandProvider
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.command_decorator import command


```
