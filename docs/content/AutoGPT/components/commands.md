# 🛠️ Commands

Commands are a way for the agent to do anything; e.g. interact with the user or APIs and use tools. They are provided by components that implement the `CommandProvider` [⚙️ Protocol](./protocols.md). Commands are functions that can be called by the agent, they can have parameters and return values that will be seen by the agent.

```py
class CommandProvider(Protocol):
    def get_commands(self) -> Iterator[Command]:
        ...
```

## `command` decorator

The easiest and recommended way to provide a command is to use `command` decorator on a component method and then just yield it in `get_commands` as part of your provider. Each command needs a name, description and a parameter schema - `JSONSchema`. By default method name is used as a command name, and first part of docstring for the description (before first double newline) and schema can be provided in the decorator.

### Example usage of `command` decorator

```py
# Assuming this is inside some component class
@command(
    parameters={
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

We can provide `names` and `description` in the decorator, the above command is equivalent to:

```py
@command(
    names=["multiply"],
    description="Multiplies two numbers.",
    parameters={
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
    def multiply_command(self, a: int, b: int) -> str:
        return str(a * b)
```

To provide the `multiply` command to the agent, we need to yield it in `get_commands`:

```py
def get_commands(self) -> Iterator[Command]:
    yield self.multiply
```

## Creating `Command` directly

If you don't want to use the decorator, you can create a `Command` object directly.

```py

def multiply(self, a: int, b: int) -> str:
        return str(a * b)

def get_commands(self) -> Iterator[Command]:
    yield Command(
        names=["multiply"],
        description="Multiplies two numbers.",
        method=self.multiply,
        parameters=[
            CommandParameter(name="a", spec=JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="The first number",
                required=True,
            )),
            CommandParameter(name="b", spec=JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="The second number",
                required=True,
            )),
        ],
    )
```