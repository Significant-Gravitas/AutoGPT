# Built-in Components

This page lists all [🧩 Components](./components.md) and [⚙️ Protocols](./protocols.md) they implement that are natively provided. They are used by the AutoGPT agent.

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

**DirectiveProvider**
- Resource information that it's possible to read and write files

**CommandProvider**
- `read_file` used to read file
- `write_file` used to write file
- `list_folder` lists all files in a folder 

## `CodeExecutorComponent`

Lets the agent execute non-interactive Shell commands and Python code. Python execution works only if Docker is available.

**CommandProvider**
- `execute_shell` execute shell command
- `execute_shell_popen` execute shell command with popen
- `execute_python_code` execute Python code
- `execute_python_file` execute Python file

## `EventHistoryComponent`

Keeps track of agent's actions and their outcomes. Provides their summary to the prompt.

**MessageProvider**
- Agent's progress summary

**AfterParse**
- Register agent's action

**ExecutionFailuer**
- Rewinds the agent's action, so it isn't saved

**AfterExecute**
- Saves the agent's action result in the history

## `GitOperationsComponent`

**CommandProvider**
- `clone_repository` used to clone a git repository

## `ImageGeneratorComponent`

Adds ability to generate images using various providers, see [Image Generation configuration](./../configuration/imagegen.md) to learn more.

**CommandProvider**
- `generate_image` used to generate an image given a prompt

## `WebSearchComponent`

Allows agent to search the web.

**DirectiveProvider**
- Resource information that it's possible to search the web

**CommandProvider**
- `search_web` used to search the web using DuckDuckGo
- `google` used to search the web using Google, requires API key

## `WebSeleniumComponent`

Allows agent to read websites using Selenium.

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