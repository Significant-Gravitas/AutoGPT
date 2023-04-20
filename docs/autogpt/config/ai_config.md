# `AIConfig` class module documentation

This module defines the `AIConfig` class, which is responsible for holding all the configuration information related to an AI.

## `AIConfig` class

The `AIConfig` class has the following attributes:
- `ai_name` (str): The name of the AI.
- `ai_role` (str): The description of the AI's role.
- `ai_goals` (list): The list of objectives the AI is supposed to complete.

The class has the following methods:

### `__init__(self, ai_name: str = "", ai_role: str = "", ai_goals: list | None = None) -> None`

Initializes an instance of `AIConfig`.

#### Parameters:
- `ai_name` (str): The name of the AI.
- `ai_role` (str): The description of the AI's role.
- `ai_goals` (list): The list of objectives the AI is supposed to complete.

#### Returns:
None.

### `load(config_file: str = SAVE_FILE) -> AIConfig`

Returns class object with parameters (ai_name, ai_role, ai_goals) loaded from yaml file if yaml file exists, else returns class with no parameters.

#### Parameters:
- `config_file` (str): The path to the config yaml file. DEFAULT: "../ai_settings.yaml"

#### Returns:
- `cls` (AIConfig object): An instance of `AIConfig` class.

### `save(self, config_file: str = SAVE_FILE) -> None`

Saves the class parameters to the specified file yaml file path as a yaml file.

#### Parameters:
- `config_file` (str): The path to the config yaml file. DEFAULT: "../ai_settings.yaml"

#### Returns:
None.

### `construct_full_prompt(self, prompt_generator: Optional[PromptGenerator] = None) -> str`

Returns a prompt to the user with the class information in an organized fashion.

#### Parameters:
- `prompt_generator` (PromptGenerator object): An optional instance of `PromptGenerator` that will be used to generate the prompt.

#### Returns:
- `full_prompt` (str): A string containing the initial prompt for the user including the ai_name, ai_role and ai_goals.