# Prompt Generator Module

This module contains a `PromptGenerator` class which can generate custom prompt strings based on constraints, commands, resources, and performance evaluations.

## `PromptGenerator` class

This class has the following methods and attributes:

### Methods

#### `__init__`

```python
def __init__(self) -> None:
```

Initialize the `PromptGenerator` object with empty lists of constraints, commands, resources, performance evaluations, and goals.

#### `add_constraint`

```python
def add_constraint(self, constraint: str) -> None:
```

Add a constraint to the constraints list.

- `constraint` - The constraint to be added.

#### `add_command`

```python
def add_command(
    self,
    command_label: str,
    command_name: str,
    args=None,
    function: Optional[Callable] = None,
) -> None:
```

Add a command to the commands list with a label, name, and optional arguments.

- `command_label` - The label of the command.
- `command_name` - The name of the command.
- `args` (optional) - A dictionary containing argument names and their values. Defaults to None.
- `function` (optional) - A callable function to be called when the command is executed. Defaults to None.

#### `_generate_command_string`

```python
def _generate_command_string(self, command: Dict[str, Any]) -> str:
```

Generate a formatted string representation of a command.

- `command` - A dictionary containing command information.

#### `add_resource`

```python
def add_resource(self, resource: str) -> None:
```

Add a resource to the resources list.

- `resource` - The resource to be added.

#### `add_performance_evaluation`

```python
def add_performance_evaluation(self, evaluation: str) -> None:
```

Add a performance evaluation item to the performance_evaluation list.

- `evaluation` - The evaluation item to be added.

#### `_generate_numbered_list`

```python
def _generate_numbered_list(self, items: List[Any], item_type="list") -> str:
```

Generate a numbered list from given items based on the item_type.

- `items` - A list of items to be numbered.
- `item_type` (optional) - The type of items in the list. Defaults to 'list'.

Returns the formatted numbered list.

#### `generate_prompt_string`

```python
def generate_prompt_string(self) -> str:
```

Generate a prompt string based on the constraints, commands, resources, and performance evaluations.

Returns the generated prompt string.

### Attributes

- `constraints` - A list of constraints.
- `commands` - A list of commands.
- `resources` - A list of resources.
- `performance_evaluation` - A list of performance evaluations.
- `goals` - A list of goals.
- `command_registry` - The registry of commands.
- `name` - The name of the generator.
- `role` - The role of the generator (e.g. "AI").
- `response_format` - A dictionary describing the format of the response. 

## Example

```python
from prompt_generator import PromptGenerator


generator = PromptGenerator()
generator.add_constraint("Must be at least 18 years old.")
generator.add_command("Greet", "greet", {"name": "Alice"})
generator.add_resource("Bicycle")
generator.add_performance_evaluation("Successful completion of task.")
prompt_string = generator.generate_prompt_string()
print(prompt_string)
```

Output:

```
Constraints:
1. "Must be at least 18 years old."

Commands:
1. "do_nothing": "do_nothing", args: {}
2. "terminate": "terminate", args: {}
3. "Greet": "greet", args: "name": "Alice"

Resources:
1. "Bicycle"

Performance Evaluation:
1. "Successful completion of task."

You should only respond in JSON format as described below 
Response Format: 
{
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {"name": "command name", "args": {"arg name": "value"}}
}
Ensure the response can be parsed by Python json.loads
```