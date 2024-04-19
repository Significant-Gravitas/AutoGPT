# Component Agents Changes

## Breaking changes

- Removed command categories and `DISABLED_COMMAND_CATEGORIES` environment variable. Use `DISABLED_COMMANDS` environment variable to disable individual commands.
- Changed `command` decorator; old-style commands are no longer supported. Implement `CommandProvider` on components instead.
- Removed `CommandRegistry`, now all commands are provided by components implementing `CommandProvider`.
- Removed `prompt_config` from `AgentSettings`.
- Removed plugin support: old plugins will no longer be loaded and executed.
- Removed `PromptScratchpad`, it was used by plugins and is no longer needed.
- Changed `ThoughtProcessOutput` from tuple to pydantic `BaseModel`.

## Other changes

- Created `AgentComponent`, protocols and logic to execute them.
- `BaseAgent` and `Agent` is now composed of components.
- Moved some logic from `BaseAgent` to `Agent`.
- Moved agent features and commands to components.
- Removed check if the same operation is about to be executed twice in a row.
- Removed file logging from `FileManagerComponent` (formerly `AgentFileManagerMixin`)
- Updated tests
- Added docs

See [Introduction](./introduction.md) for more information.