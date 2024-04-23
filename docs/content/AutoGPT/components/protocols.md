# ⚙️ Protocols

Protocols are *interfaces* implemented by [Components](./components.md) used to group related functionality. Each protocol needs to be handled explicitly by the agent at some point of the execution. We provide a comprehensive list of built-in protocols that are already handled in the built-in `Agent`, so when you inherit from the base agent all built-in protocols will work!

**Protocols are listed in the order of the default execution.**

## Order-independent protocols

Components implementing exclusively order-independent protocols can added in any order, including in-between ordered protocols.

### `DirectiveProvider`

Yields constraints, resources and best practices for the agent. This has no direct impact on other protocols; is purely informational and will be passed to a llm when the prompt is built.

```py
class DirectiveProvider(AgentComponent):
    def get_constraints(self) -> Iterator[str]:
        return iter([])

    def get_resources(self) -> Iterator[str]:
        return iter([])

    def get_best_practices(self) -> Iterator[str]:
        return iter([])
```

**Example** A web-search component can provide a resource information. Keep in mind that this actually doesn't allow the agent to access the internet. To do this a relevant `Command` needs to be provided.

```py
class WebSearchComponent(DirectiveProvider):
    def get_resources(self) -> Iterator[str]:
        yield "Internet access for searches and information gathering."
    # We can skip "get_constraints" and "get_best_practices" if they aren't needed
```

### `CommandProvider`

Provides a command that can be executed by the agent.

```py
class CommandProvider(AgentComponent):
    def get_commands(self) -> Iterator[Command]:
        ...
```

The easiest way to provide a command is to use `command` decorator on a component method and then yield the method. Each command needs a name, description and a parameter schema using `JSONSchema`. By default method name is used as a command name, and first part of docstring for the description (before `Args:` or `Returns:`) and schema can be provided in the decorator.

**Example** Calculator component that can perform multiplication. Agent is able to call this command if it's relevant to a current task and will see the returned result.

```py
from autogpt.agents.components import Component
from autogpt.agents.protocols import CommandProvider
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.utils.command_decorator import command


class CalculatorComponent(CommandProvider):
    get_commands(self) -> Iterator[Command]:
        yield self.multiply

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

The order of components implementing order-dependent protocols is important.
Some components may depend on the results of components before them.

### `MessageProvider`

Yields messages that will be added to the agent's prompt. You can use either `ChatMessage.user()`: this will interpreted as a user-sent message or `ChatMessage.system()`: that will be more important.

```py
class MessageProvider(AgentComponent):
    def get_messages(self) -> Iterator[ChatMessage]:
        ...
```

**Example** Component that provides a message to the agent's prompt.

```py
class HelloComponent(MessageProvider):
    def get_messages(self) -> Iterator[ChatMessage]:
        yield ChatMessage.user("Hello World!")
```

### `AfterParse`

Protocol called after the response is parsed.

```py
class AfterParse(AgentComponent):
    def after_parse(self, response: ThoughtProcessOutput) -> None:
        ...
```

**Example** Component that logs the response after it's parsed.

```py
class LoggerComponent(AfterParse):
    def after_parse(self, response: ThoughtProcessOutput) -> None:
        logger.info(f"Response: {response}")
```

### `ExecutionFailure` 

Protocol called when the execution of the command fails.

```py
class ExecutionFailure(AgentComponent):
    @abstractmethod
    def execution_failure(self, error: Exception) -> None:
        ...
```

**Example** Component that logs the error when the command fails.

```py
class LoggerComponent(ExecutionFailure):
    def execution_failure(self, error: Exception) -> None:
        logger.error(f"Command execution failed: {error}")
```

### `AfterExecute`

Protocol called after the command is successfully executed by the agent.

```py
class AfterExecute(AgentComponent):
    def after_execute(self, result: ActionResult) -> None:
        ...
```

**Example** Component that logs the result after the command is executed.

```py
class LoggerComponent(AfterExecute):
    def after_execute(self, result: ActionResult) -> None:
        logger.info(f"Result: {result}")
```