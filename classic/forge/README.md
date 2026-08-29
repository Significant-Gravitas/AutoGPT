# AutoGPT Forge

Core autonomous agent framework for building AI agents.

## Quick Start

All commands run from the `classic/` directory (parent of this directory):

```bash
# Install (one-time setup)
cd classic
poetry install

# Configure
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run
poetry run python -m forge
```

The agent server runs on `http://localhost:8000` by default.

## Configuration

### Environment Variables (`.env`)

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional LLM settings
SMART_LLM=gpt-4o                    # Model for complex reasoning
FAST_LLM=gpt-4o-mini                # Model for simple tasks
EMBEDDING_MODEL=text-embedding-3-small

# Optional search providers
TAVILY_API_KEY=tvly-...
SERPER_API_KEY=...
GOOGLE_API_KEY=...
GOOGLE_CUSTOM_SEARCH_ENGINE_ID=...

# Optional infrastructure
LOG_LEVEL=DEBUG                     # DEBUG, INFO, WARNING, ERROR
DATABASE_STRING=sqlite:///agent.db  # Agent Protocol database
PORT=8000                           # Server port
FILE_STORAGE_BACKEND=local          # local, s3, or gcs
```

### Workspace Settings (`.autogpt/autogpt.yaml`)

Workspace-wide permissions for all agents:

```yaml
allow:
  - read_file({workspace}/**)
  - write_to_file({workspace}/**)
  - list_folder({workspace}/**)
  - web_search(*)

deny:
  - read_file(**.env)
  - read_file(**.key)
  - execute_shell(rm -rf:*)
  - execute_shell(sudo:*)
```

### Agent Settings (`.autogpt/agents/{id}/permissions.yaml`)

Agent-specific permission overrides:

```yaml
allow:
  - execute_python(*)
deny:
  - execute_shell(*)
```

## Workspace Structure

```
{workspace}/
├── .autogpt/
│   ├── autogpt.yaml              # Workspace permissions
│   ├── ap_server.db              # Agent Protocol database
│   └── agents/
│       └── AutoGPT-{agent_id}/
│           ├── state.json        # Agent state
│           ├── permissions.yaml  # Agent permissions
│           └── workspace/        # Agent's working directory
```

## Permissions

Permission checks follow this order (first match wins):

1. Agent deny list → Block
2. Workspace deny list → Block
3. Agent allow list → Allow
4. Workspace allow list → Allow
5. Prompt user → Interactive approval

### Pattern Syntax

Format: `command_name(glob_pattern)`

| Pattern | Description |
|---------|-------------|
| `read_file({workspace}/**)` | Read any file in workspace |
| `execute_shell(python:**)` | Execute Python commands |
| `web_search(*)` | All web searches |

Special tokens:
- `{workspace}` - Replaced with workspace path
- `**` - Matches any path including `/`
- `*` - Matches any characters except `/`

## Tutorials

The [tutorial series](https://aiedge.medium.com/autogpt-forge-e3de53cc58ec) guides you through building a custom agent:

1. [A Comprehensive Guide to Your First Steps](https://aiedge.medium.com/autogpt-forge-a-comprehensive-guide-to-your-first-steps-a1dfdf46e3b4)
2. [The Blueprint of an AI Agent](https://aiedge.medium.com/autogpt-forge-the-blueprint-of-an-ai-agent-75cd72ffde6)
3. [Interacting with your Agent](https://aiedge.medium.com/autogpt-forge-interacting-with-your-agent-1214561b06b)
4. [Crafting Intelligent Agent Logic](https://medium.com/@aiedge/autogpt-forge-crafting-intelligent-agent-logic-bc5197b14cb4)


