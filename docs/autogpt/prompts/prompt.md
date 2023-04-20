## `autogpt.setup` module

This module contains functions for setting up and initializing the AutoGPT AI assistant.

### `build_default_prompt_generator() -> PromptGenerator`

This function generates a default prompt string with various constraints, commands, resources, and performance evaluations. It returns the generated prompt string as a `PromptGenerator` object.

__Arguments__:
None

__Returns__:
- `PromptGenerator`: The generated prompt string.

__Example__:
```python
from autogpt.setup import build_default_prompt_generator

prompt_generator = build_default_prompt_generator()
```

### `construct_main_ai_config() -> AIConfig`

This function constructs the prompt for the AI to respond to.

__Arguments__:
None

__Returns__:
- `AIConfig`: The AI configuration object loaded from a configuration file.

__Example__:
```python
from autogpt.setup import construct_main_ai_config

config = construct_main_ai_config()
```