# Command-Line Interface

This module contains a `CommandRegistry` class that provides a way to manage a collection of `Command` objects. A `Command` object contains a `name` (string) and a `method` (callable), which will be executed when called upon.

## CommandRegistry

### `CommandRegistry() -> CommandRegistry`

Class constructor that initializes an empty `command registry` object.

### `register(cmd: Command) -> None`

Registers a `Command` object into the `command registry`.

### `unregister(command_name: str) -> None`

Unregisters a `Command` object by its `command name` from the `command registry`.

### `reload_commands() -> None`

Reloads all loaded commands inside the registry.

### `get_command(name: str) -> Callable[..., Any]`

Returns the `Command` method by its name.

### `call(command_name: str, **kwargs) -> Any`

Calls and returns the result of the `Command` method by its `name`, with parameters `**kwargs`.

### `command_prompt() -> str`

Returns the string representation of all registered `Command` objects.

### `import_commands(module_name: str) -> None`

Imports a module containing command plugins. This method registers any functions or classes that are decorated with the `AUTO_GPT_COMMAND_IDENTIFIER` keyword as `Command` objects.

## Command

### `Command(name: str, description: str, method: Callable[..., Any], signature: str = '', enabled: bool = True, disabled_reason: Optional[str] = None) -> Command`

Class constructor that initializes a `Command` object. A `Command` object contains a `name` (string), `description` (string), `method` (callable), `signature` (string), `enabled` (boolean), and `disabled_reason` (optional string).

### `__call__(*args, **kwargs) -> Any`

Returns the result of the `Command` method with passed arguments if the `Command` is enabled. Otherwise, returns a message indicating that the `Command` is disabled.

### `__str__() -> str`

Returns the string representation of the `Command` object with `name`, `description`, and `arguments`.