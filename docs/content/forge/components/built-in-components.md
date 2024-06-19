# Built-in Components

This page lists all [🧩 Components](./components.md) and [⚙️ Protocols](./protocols.md) they implement that are natively provided. They are used by the AutoGPT agent.
Some components have additional configuration options listed in the table, see [Component configuration](./components.md/#ordering-components) to learn more.

!!! note
    If a configuration field uses environment variable, it still can be passed using configuration model. **Value from the configuration takes precedence over env var!** Env var will be only applied if value in the configuration is not set.

## `SystemComponent`

Essential component to allow an agent to finish.

**DirectiveProvider**

- Constraints about API budget
  
**MessageProvider**

- Current time and date
- Remaining API budget and warnings if budget is low

**CommandProvider**

- `finish` used when task is completed

## `UserInteractionComponent`

Adds ability to interact with user in CLI.

**CommandProvider**

- `ask_user` used to ask user for input

## `FileManagerComponent`

Adds ability to read and write persistent files to local storage, Google Cloud Storage or Amazon's S3.
Necessary for saving and loading agent's state (preserving session).

| Config variable  | Details                                | Type  | Default                            |
| ---------------- | -------------------------------------- | ----- | ---------------------------------- |
| `files_path`     | Path to agent files, e.g. state        | `str` | `agents/{agent_id}/`[^1]           |
| `workspace_path` | Path to files that agent has access to | `str` | `agents/{agent_id}/workspace/`[^1] |

[^1] This option is set dynamically during component construction as opposed to by default inside the configuration model, `{agent_id}` is replaced with the agent's unique identifier.

**DirectiveProvider**

- Resource information that it's possible to read and write files

**CommandProvider**

- `read_file` used to read file
- `write_file` used to write file
- `list_folder` lists all files in a folder 

## `CodeExecutorComponent`

Lets the agent execute non-interactive Shell commands and Python code. Python execution works only if Docker is available.

| Config variable          | Details                                              | Type                        | Default           |
| ------------------------ | ---------------------------------------------------- | --------------------------- | ----------------- |
| `execute_local_commands` | Enable shell command execution                       | `bool`                      | `False`           |
| `shell_command_control`  | Controls which list is used                          | `"allowlist" \| "denylist"` | `"allowlist"`     |
| `shell_allowlist`        | List of allowed shell commands                       | `List[str]`                 | `[]`              |
| `shell_denylist`         | List of prohibited shell commands                    | `List[str]`                 | `[]`              |
| `docker_container_name`  | Name of the Docker container used for code execution | `str`                       | `"agent_sandbox"` |

**CommandProvider**

- `execute_shell` execute shell command
- `execute_shell_popen` execute shell command with popen
- `execute_python_code` execute Python code
- `execute_python_file` execute Python file

## `ActionHistoryComponent`

Keeps track of agent's actions and their outcomes. Provides their summary to the prompt.

| Config variable        | Details                                                 | Type        | Default            |
| ---------------------- | ------------------------------------------------------- | ----------- | ------------------ |
| `model_name`           | Name of the llm model used to compress the history      | `ModelName` | `"gpt-3.5-turbo"`  |
| `max_tokens`           | Maximum number of tokens to use for the history summary | `int`       | `1024`             |
| `spacy_language_model` | Language model used for summary chunking using spacy    | `str`       | `"en_core_web_sm"` |

**MessageProvider**

- Agent's progress summary

**AfterParse**

- Register agent's action

**ExecutionFailure**

- Rewinds the agent's action, so it isn't saved

**AfterExecute**

- Saves the agent's action result in the history

## `GitOperationsComponent`

Adds ability to iteract with git repositories and GitHub.

| Config variable   | Details                                   | Type  | Default |
| ----------------- | ----------------------------------------- | ----- | ------- |
| `github_username` | GitHub username, *ENV:* `GITHUB_USERNAME` | `str` | `None`  |
| `github_api_key`  | GitHub API key, *ENV:* `GITHUB_API_KEY`   | `str` | `None`  |

**CommandProvider**

- `clone_repository` used to clone a git repository

## `ImageGeneratorComponent`

Adds ability to generate images using various providers.

### Hugging Face

To use text-to-image models from Hugging Face, you need a Hugging Face API token.
Link to the appropriate settings page: [Hugging Face > Settings > Tokens](https://huggingface.co/settings/tokens)

### Stable Diffusion WebUI

It is possible to use your own self-hosted Stable Diffusion WebUI with AutoGPT. **Make sure you are running WebUI with `--api` enabled.**

| Config variable           | Details                                                       | Type                                    | Default                           |
| ------------------------- | ------------------------------------------------------------- | --------------------------------------- | --------------------------------- |
| `image_provider`          | Image generation provider                                     | `"dalle" \| "huggingface" \| "sdwebui"` | `"dalle"`                         |
| `huggingface_image_model` | Hugging Face image model, see [available models]              | `str`                                   | `"CompVis/stable-diffusion-v1-4"` |
| `huggingface_api_token`   | Hugging Face API token, *ENV:* `HUGGINGFACE_API_TOKEN`        | `str`                                   | `None`                            |
| `sd_webui_url`            | URL to self-hosted Stable Diffusion WebUI                     | `str`                                   | `"http://localhost:7860"`         |
| `sd_webui_auth`           | Basic auth for Stable Diffusion WebUI, *ENV:* `SD_WEBUI_AUTH` | `str` of format `{username}:{password}` | `None`                            |

[available models]: https://huggingface.co/models?pipeline_tag=text-to-image

**CommandProvider**

- `generate_image` used to generate an image given a prompt

## `WebSearchComponent`

Allows agent to search the web. Google credentials aren't required for DuckDuckGo. [Instructions how to set up Google API key](../../AutoGPT/configuration/search.md)

| Config variable                  | Details                                                                 | Type  | Default |
| -------------------------------- | ----------------------------------------------------------------------- | ----- | ------- |
| `google_api_key`                 | Google API key, *ENV:* `GOOGLE_API_KEY`                                 | `str` | `None`  |
| `google_custom_search_engine_id` | Google Custom Search Engine ID, *ENV:* `GOOGLE_CUSTOM_SEARCH_ENGINE_ID` | `str` | `None`  |
| `duckduckgo_max_attempts`        | Maximum number of attempts to search using DuckDuckGo                   | `int` | `3`     |

**DirectiveProvider**

- Resource information that it's possible to search the web

**CommandProvider**

- `search_web` used to search the web using DuckDuckGo
- `google` used to search the web using Google, requires API key

## `WebSeleniumComponent`

Allows agent to read websites using Selenium.

| Config variable               | Details                                     | Type                                          | Default                                                                                                                      |
| ----------------------------- | ------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `model_name`                  | Name of the llm model used to read websites | `ModelName`                                   | `"gpt-3.5-turbo"`                                                                                                            |
| `web_browser`                 | Web browser used by Selenium                | `"chrome" \| "firefox" \| "safari" \| "edge"` | `"chrome"`                                                                                                                   |
| `headless`                    | Run browser in headless mode                | `bool`                                        | `True`                                                                                                                       |
| `user_agent`                  | User agent used by the browser              | `str`                                         | `"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"` |
| `browse_spacy_language_model` | Spacy language model used for chunking text | `str`                                         | `"en_core_web_sm"`                                                                                                           |

**DirectiveProvider**

- Resource information that it's possible to read websites

**CommandProvider**

- `read_website` used to read a specific url and look for specific topics or answer a question

## `ContextComponent`

Adds ability to keep up-to-date file and folder content in the prompt.

**MessageProvider**

- Content of elements in the context

**CommandProvider**

- `open_file` used to open a file into context
- `open_folder` used to open a folder into context
- `close_context_item` remove an item from the context

## `WatchdogComponent`

Watches if agent is looping and switches to smart mode if necessary.

**AfterParse**

- Investigates what happened and switches to smart mode if necessary
