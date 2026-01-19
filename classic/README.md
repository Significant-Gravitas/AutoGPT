# AutoGPT Classic

AutoGPT Classic was an experimental project to demonstrate autonomous GPT-4 operation. It was designed to make GPT-4 independently operate and chain together tasks to achieve more complex goals.

## Project Status

**This project is unsupported, and dependencies will not be updated.** It was an experiment that has concluded its initial research phase. If you want to use AutoGPT, you should use the [AutoGPT Platform](/autogpt_platform).

For those interested in autonomous AI agents, we recommend exploring more actively maintained alternatives or referring to this codebase for educational purposes only.

## Overview

AutoGPT Classic was one of the first implementations of autonomous AI agents - AI systems that can independently:
- Break down complex goals into smaller tasks
- Execute those tasks using available tools and APIs
- Learn from the results and adjust its approach
- Chain multiple actions together to achieve an objective

## Structure

- `/benchmark` - Performance testing tools
- `/forge` - Core autonomous agent framework
- `/original_autogpt` - Original implementation

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation

```bash
# Clone the repository
git clone https://github.com/Significant-Gravitas/AutoGPT.git
cd classic

# Install forge (core library)
cd forge && poetry install

# Or install original_autogpt (includes forge as dependency)
cd original_autogpt && poetry install

# Install benchmark (optional)
cd benchmark && poetry install
```

### Configuration

Configuration uses a layered system:

1. **Environment variables** (`.env` file)
2. **Workspace settings** (`.autogpt/autogpt.yaml`)
3. **Agent settings** (`.autogpt/agents/{id}/permissions.yaml`)

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Key environment variables:
```bash
# Required
OPENAI_API_KEY=sk-...

# Optional LLM settings
SMART_LLM=gpt-4o                    # Model for complex reasoning
FAST_LLM=gpt-4o-mini                # Model for simple tasks

# Optional search providers
TAVILY_API_KEY=tvly-...
SERPER_API_KEY=...

# Optional infrastructure
LOG_LEVEL=DEBUG
PORT=8000
FILE_STORAGE_BACKEND=local          # local, s3, or gcs
```

### Running

```bash
# Run forge agent
cd forge && poetry run python -m forge

# Run original autogpt server
cd original_autogpt && poetry run serve --debug

# Run autogpt CLI
cd original_autogpt && poetry run autogpt
```

Agents run on `http://localhost:8000` by default.

### Benchmarking

```bash
cd benchmark && poetry run agbenchmark
```

### Testing

```bash
cd forge && poetry run pytest
cd original_autogpt && poetry run pytest
```

## Workspaces

Agents operate within a **workspace** directory that contains all agent data and files:

```
{workspace}/
├── .autogpt/
│   ├── autogpt.yaml              # Workspace-level permissions
│   ├── ap_server.db              # Agent Protocol database (server mode)
│   └── agents/
│       └── AutoGPT-{agent_id}/
│           ├── state.json        # Agent profile, directives, history
│           ├── permissions.yaml  # Agent-specific permissions
│           └── workspace/        # Agent's sandboxed working directory
```

- The workspace defaults to the current working directory
- Multiple agents can coexist in the same workspace
- Agent file access is sandboxed to their `workspace/` subdirectory
- State persists across sessions via `state.json`

## Permissions

AutoGPT uses a **layered permission system** with pattern matching:

### Permission Files

| File | Scope | Location |
|------|-------|----------|
| `autogpt.yaml` | All agents in workspace | `.autogpt/autogpt.yaml` |
| `permissions.yaml` | Single agent | `.autogpt/agents/{id}/permissions.yaml` |

### Permission Format

```yaml
allow:
  - read_file({workspace}/**)     # Read any file in workspace
  - write_to_file({workspace}/**) # Write any file in workspace
  - web_search(*)                 # All web searches

deny:
  - read_file(**.env)             # Block .env files
  - execute_shell(sudo:*)         # Block sudo commands
```

### Check Order (First Match Wins)

1. Agent deny → Block
2. Workspace deny → Block
3. Agent allow → Allow
4. Workspace allow → Allow
5. Prompt user → Interactive approval

### Interactive Approval

When prompted, users can approve commands with different scopes:
- **Once** - Allow this one time only
- **Agent** - Always allow for this agent
- **Workspace** - Always allow for all agents
- **Deny** - Block this command

### Default Security

Denied by default:
- Sensitive files (`.env`, `.key`, `.pem`)
- Destructive commands (`rm -rf`, `sudo`)
- Operations outside the workspace

## Security Notice

This codebase has **known vulnerabilities** and issues with its dependencies. It will not be updated to new dependencies. Use for educational purposes only.

## License

This project segment is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

Please refer to the [documentation](https://docs.agpt.co) for more detailed information about the project's architecture and concepts.
