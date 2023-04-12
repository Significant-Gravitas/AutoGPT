# Auto-GPT

The `auto_gpt` module contains the code for the Auto-GPT AI assistant. This module imports and uses several other custom modules such as `commands`, `utils`, `memory`, `data`, `chat`, `spinner`, `speak`, `config`, `json_parser`, `ai_config`, and the third-party package `colorama`. 

Before running the AI assistant, it is important to set your OpenAI API key in the `config.py` file or as an environment variable. This can be obtained from [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys)

## Usage

```python
python auto_gpt.py
```

## Getting Started

The `load_variables` function in `auto_gpt.py` will prompt the user to enter a name for the AI, its role, and up to 5 goals. If a `config.yaml` file exists in the project's root directory, it will load these values from there instead. If any of these values are left blank, default values are assigned.

The `parse_arguments` function is used to parse command line arguments. Pass the `--continuous` flag to enable 'Continuous Mode', the `--speak` flag to enable 'Speak Mode', the `--debug` flag to enable debug mode, and the `--gpt3only` flag to enable GPT-3.5 only mode.

The AI assistant then enters a loop where it prompts the user to enter a command, sends the message to the GPT-3 API to get the assistant's response, then executes the resulting command.

## Functions and Classes

### `load_variables(config_file="config.yaml"):`

Loads variables from a `config.yaml` file if it exists, otherwise prompts the user for input. The user is prompted to enter a name for the AI, its role, and up to 5 goals. 

### `configure_logging():`

Configures logging settings for the `AutoGPT` logger.

### `check_openai_api_key():`

Checks if the OpenAI API key is set in the `config.py` file or as an environment variable.

### `print_to_console(title, title_color, content, speak_text=False,min_typing_speed=0.05,max_typing_speed=0.01):`

Prints text to the console with a typing effect. `title` is the title of the message, `title_color` is the color of the title, `content` is the message to print, `speak_text` is a boolean flag to enable text-to-speech, and `min_typing_speed` and `max_typing_speed` are the minimum and maximum typing speeds, respectively.

### `print_assistant_thoughts(assistant_reply):`

Prints the assistant's thoughts to the console. `assistant_reply` is the response from the assistant.

### `load_variables(config_file="config.yaml"):`

Loads variables for Auto-GPT from `config.yaml` if it exists, otherwise prompts the user for input.

### `construct_prompt():`

Constructs the prompt for