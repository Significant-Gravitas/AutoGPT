# Command and Control

This module is responsible for executing and managing commands received from the AI. It exports several functions for executing commands, parsing their arguments, and listing available commands. It also contains several helper functions for translating command names and checking argument validity.

## Functions

### is_valid_int

```python
def is_valid_int(value: str) -> bool:
```

This function takes a string value and checks if it is a valid integer. It returns True if the value is a valid integer, and False otherwise.

### get_command

```python
def get_command(response_json: Dict) -> Tuple[str, Dict]:
```

This function takes a dictionary object (in JSON format) and parses it to extract the command name and arguments. It returns a tuple with the command name and a dictionary of arguments.

### map_command_synonyms

```python
def map_command_synonyms(command_name: str) -> str:
```

This function takes a command name and checks if it matches any known hallucinations. It returns the actual command name that should be used.

### execute_command

```python
def execute_command(command_registry: CommandRegistry, command_name: str, arguments, prompt: PromptGenerator) -> str:
```

This function takes a command registry, command name, arguments dictionary, and prompt generator. It executes the given command with the given arguments, and returns a string result.

### get_text_summary

```python
@command("get_text_summary", "Get text summary", '"url": "<url>", "question": "<question>"')
def get_text_summary(url: str, question: str) -> str:
```

This function takes a URL and question, and returns a summary of the text found at the URL. This function is decorated with the `@command` decorator, which registers it as a valid command and provides metadata about the command.

### get_hyperlinks

```python
@command("get_hyperlinks", "Get text summary", '"url": "<url>"')
def get_hyperlinks(url: str) -> Union[str, List[str]]:
```

This function takes a URL and returns a list of hyperlinks found at the URL. This function is decorated with the `@command` decorator, which registers it as a valid command and provides metadata about the command.

### shutdown

```python
def shutdown() -> NoReturn:
```

This function simply prints a message that the program is shutting down, and calls the `quit()` function to exit the program.

### start_agent

```python
@command("start_agent", "Start GPT Agent", '"name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"')
def start_agent(name: str, task: str, prompt: str, model=CFG.fast_llm_model) -> str:
```

This function creates a new agent with the given name, task, and prompt. It returns a string message indicating the status of the agent creation. This function is decorated with the `@command` decorator, which registers it as a valid command and provides metadata about the command.

### message_agent

```python
@command("message_agent", "Message GPT Agent", '"key": "<key>", "message": "<message>"')
def message_agent(key: str, message: str) -> str:
```

This function sends a message to an existing GPT agent with the given key. It returns a string message indicating the response received from the agent. This function is decorated with the `@command` decorator, which registers it as a valid command and provides metadata about the command.

### list_agents

```python
@command("list_agents", "List GPT Agents", "")
def list_agents() -> str:
```

This function lists all existing GPT agents. It returns a string message containing a list of all agents. This function is decorated with the `@command` decorator, which registers it as a valid command and provides metadata about the command.

### delete_agent

```python
@command("delete_agent", "Delete GPT Agent", '"key": "<key>"')
def delete_agent(key: str) -> str:
```

This function deletes a GPT agent with the given key. It returns a string message indicating whether the agent was successfully deleted or not. This function is decorated with the `@command` decorator, which registers it as a valid command and provides metadata about the command.