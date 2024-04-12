# 🛠️ Commands

Commands are a way for the agent to do anything; e.g. intercting with user or APIs and using tools. They are provided by components that implement the `CommandProvider` protocol.

```py
class CommandProvider(Protocol):
    def get_commands(self) -> Iterator[Command]:
        ...
```

## `command` decorator

The easiest way to provide a command is to use `command` decorator on a component method and then just yield it. Each command needs a name, description and a parameter schema - `JSONSchema`. By default method name is used as a command name, and first part of docstring for the description (before first double newline) and schema can be provided in the decorator.

### Example usage of `command` decorator

```py
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
