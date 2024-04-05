# ⚙️ Protocols

Protocols are *interfaces* implemented by [Components](./components.md) used to group related functionality. Each protocol needs to be handled explicitly by the agent at some point of the execution. We provide a comprehensive list of built-in protocols that are already handled in the built-in `Agent`, so when you inherit from the base agent all built-in protocols will work!

**Protocols are listed in the order of the default execution.**

## Order-independent protocols

Components implementing exclusively order-independent protocols can added in any order, including in-between ordered protocols.

### `DirectiveProvider`

Yields constraints, resources and best practices for the agent. This is purely informational and will be passed to a llm after prompt is ready using `BuildPrompt` protocol.

```py
class DirectiveProvider(Protocol):
    def get_contraints(self) -> Iterator[str]:
        return iter([])

    def get_resources(self) -> Iterator[str]:
        return iter([])

    def get_best_practices(self) -> Iterator[str]:
        return iter([])
```

**Example** A web-search component can provide a resource information. Keep in mind that this actually doesn't allow the agent to access the internet. To do this a relevant `Command` needs to be provided.

```py
class WebSearchComponent(Component, DirectiveProvider):
    def get_resources(self) -> Iterator[str]:
        yield "Internet access for searches and information gathering."
    # We can skip "get_constraints" and "get_best_practices" if they aren't needed
```

### `CommandProvider`

Provides a command that can be executed by the agent.

```py
class CommandProvider(Protocol):
    def get_commands(self) -> Iterator[Command]:
        ...
```

The easiest way to provide a command is to use `command` decorator on a component method and then yield `Command.from_decorated_function(...)`. Each command needs a name, description and a parameter schema using `JSONSchema`. By default method name is used as a command name, and first part of docstring for the description (before `Args:` or `Returns:`) and schema can be provided in the decorator.

**Example** Calculator component that can perform multiplication. Agent is able to call this command if it's relevant to a current task and will see the returned result.

```py
from autogpt.agents.components import Component
from autogpt.agents.protocols import CommandProvider
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.command_decorator import command


class CalculatorComponent(Component, CommandProvider):
    get_commands(self) -> Iterator[Command]:
        yield Command.from_decorated_function(self.add)

    @command(parameters={
            "a": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="The first number",
                required=True,
            ),
            "b": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="The second number",
                required=True,
            )})
    def multiply(self, a: int, b: int) -> str:
        """
        Multiplies two numbers.
        
        Args:
            a: First number
            b: Second number

        Returns:
            Result of multiplication
        """
        return str(a * b)
```

The agent will be able to call this command, named `multiply` with two arguments and will receive the result. The command description will be: `Multiplies two numbers.`

To learn more about commands see [🛠️ Commands](./commands.md).

## Order-dependent protocols

The order of components implementing order-dependent protocols is important because the latter ones may depend on the results of the former ones.

### `MessageProvider`

Yields messages that will be added to the agent's prompt. You can use either `ChatMessage.user()`: this will interpreted as a user-sent message or `ChatMessage.system()`: that will be more important.

```py
class MessageProvider(Protocol):
    def get_messages(self) -> Iterator[ChatMessage]:
        ...
```

**Example** Component that provides a message to the agent's prompt.

```py
class HelloComponent(Component, MessageProvider):
    def get_messages(self) -> Iterator[ChatMessage]:
        yield ChatMessage.user("Hello World!")
```

### `BuildPrompt`

Is responsible to connect messages, commands and directives to the agent's prompt that is ready to be sent to a llm. There usually is only one component implementing this protocol.
The result of this protocol is a `ChatPrompt` object wrapped inside `Single` for architectural reasons. This may change in the future.

```py
class BuildPrompt(Protocol):
    def build_prompt(self, messages: List[ChatMessage], commands: List[Command], directives: List[str]) -> Single[ChatPrompt]:
        ...
```

**Example** Component that builds a prompt from messages, commands and directives.

```py
class PromptBuilderComponent(Component, BuildPrompt):
    def build_prompt(self, messages: List[ChatMessage], commands: List[Command], task: str, profile: AIProfile, directives: AIDirectives) -> Single[ChatPrompt]:
        messages.insert(
            0,
            ChatMessage.system(
                f"You are {profile.ai_name}, {profile.ai_role.rstrip('.')}."
                "## Constraints\n"
                f"{format_numbered_list(directives.constraints)}\n"
                "## Resources\n"
                f"{format_numbered_list(directives.resources)}\n"
                "## Best practices\n"
                f"{format_numbered_list(directives.best_practices)}\n"
            ),
        )
        messages.insert(1, ChatMessage.user(f'"""{task}"""'))
        return Single(ChatPrompt(messages=messages, functions=commands))
```

### `ParseResponse`

<!-- TODO kcze -->
*Depracated*

### `AfterParse`

Protocol called after the response is parsed.

```py
class AfterParse(Protocol):
    def after_parse(self, response: ThoughtProcessOutput) -> None:
        ...
```

**Example** Component that logs the response after it's parsed.

```py
class LoggerComponent(Component, AfterParse):
    def after_parse(self, response: ThoughtProcessOutput) -> None:
        logger.info(f"Response: {response}")
```

### `AfterExecute`

Protocol called after the command is executed by the agent.

```py
class AfterExecute(Protocol):
    def after_execute(self, result: ActionResult) -> None:
        ...
```

**Example** Component that logs the result after the command is executed.

```py
class LoggerComponent(Component, AfterExecute):
    def after_execute(self, result: ActionResult) -> None:
        logger.info(f"Result: {result}")
```