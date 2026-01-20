# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoGPT Classic is an experimental, **unsupported** project demonstrating autonomous GPT-4 operation. Dependencies will not be updated, and the codebase contains known vulnerabilities. This is preserved for educational/historical purposes.

## Repository Structure

```
classic/
├── pyproject.toml          # Single consolidated Poetry project
├── poetry.lock             # Single lock file
├── forge/
│   └── forge/              # Core agent framework package
├── original_autogpt/
│   └── autogpt/            # AutoGPT agent package
├── direct_benchmark/
│   └── direct_benchmark/   # Benchmark harness package
└── benchmark/              # Challenge definitions (data, not code)
```

All packages are managed by a single `pyproject.toml` at the classic/ root.

## Common Commands

### Setup & Install
```bash
# Install everything from classic/ directory
cd classic
poetry install
```

### Running Agents
```bash
# Run forge agent
poetry run python -m forge

# Run original autogpt server
poetry run serve --debug

# Run autogpt CLI
poetry run autogpt
```

Agents run on `http://localhost:8000` by default.

### Benchmarking
```bash
# Run benchmarks
poetry run direct-benchmark run

# Run specific strategies and models
poetry run direct-benchmark run \
    --strategies one_shot,rewoo \
    --models claude \
    --parallel 4

# Run a single test
poetry run direct-benchmark run --tests ReadFile

# List available commands
poetry run direct-benchmark --help
```

### Testing
```bash
poetry run pytest                              # All tests
poetry run pytest forge/tests/                 # Forge tests only
poetry run pytest original_autogpt/tests/      # AutoGPT tests only
poetry run pytest -k test_name                 # Single test by name
poetry run pytest path/to/test.py              # Specific test file
poetry run pytest --cov                        # With coverage
```

### Linting & Formatting

Run from the classic/ directory:

```bash
# Format everything (recommended to run together)
poetry run black . && poetry run isort .

# Check formatting (CI-style, no changes)
poetry run black --check . && poetry run isort --check-only .

# Lint
poetry run flake8        # Style linting

# Type check
poetry run pyright       # Type checking (some errors are expected in infrastructure code)
```

Note: Always run linters over the entire directory, not specific files, for best results.

## Architecture

### Forge (Core Framework)
The `forge` package is the foundation that other components depend on:
- `forge/agent/` - Agent implementation and protocols
- `forge/llm/` - Multi-provider LLM integrations (OpenAI, Anthropic, Groq, LiteLLM)
- `forge/components/` - Reusable agent components
- `forge/file_storage/` - File system abstraction
- `forge/config/` - Configuration management

### Original AutoGPT
- `original_autogpt/autogpt/app/` - CLI application entry points
- `original_autogpt/autogpt/agents/` - Agent implementations
- `original_autogpt/autogpt/agent_factory/` - Agent creation logic

### Direct Benchmark
Benchmark harness for testing agent performance:
- `direct_benchmark/direct_benchmark/` - CLI and harness code
- `benchmark/agbenchmark/challenges/` - Test cases organized by category (code, retrieval, data, etc.)
- Reports generated in `direct_benchmark/reports/`

### Package Structure
All three packages are included in a single Poetry project. Imports are fully qualified:
- `from forge.agent.base import BaseAgent`
- `from autogpt.agents.agent import Agent`
- `from direct_benchmark.harness import BenchmarkHarness`

## Code Style

- Python 3.12 target
- Line length: 88 characters (Black default)
- Black for formatting, isort for imports (profile="black")
- Type hints with Pyright checking

## Testing Patterns

- Async support via pytest-asyncio
- Fixtures defined in `conftest.py` files provide: `tmp_project_root`, `storage`, `config`, `llm_provider`, `agent`
- Tests requiring API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY) will skip if not set

## Environment Setup

Copy `.env.example` to `.env` in the relevant directory and add your API keys:
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, etc.
```

## Workspaces

Agents operate within a **workspace** - a directory containing all agent data and files. The workspace root defaults to the current working directory.

### Workspace Structure

```
{workspace}/
├── .autogpt/
│   ├── autogpt.yaml              # Workspace-level permissions
│   ├── ap_server.db              # Agent Protocol database (server mode)
│   └── agents/
│       └── AutoGPT-{agent_id}/
│           ├── state.json        # Agent profile, directives, action history
│           ├── permissions.yaml  # Agent-specific permission overrides
│           └── workspace/        # Agent's sandboxed working directory
```

### Key Concepts

- **Multiple agents** can coexist in the same workspace (each gets its own subdirectory)
- **File access** is sandboxed to the agent's `workspace/` directory by default
- **State persistence** - agent state saves to `state.json` and survives across sessions
- **Storage backends** - supports local filesystem, S3, and GCS (via `FILE_STORAGE_BACKEND` env var)

### Specifying a Workspace

```bash
# Default: uses current directory
cd /path/to/my/project && poetry run autogpt

# Or specify explicitly via CLI (if supported)
poetry run autogpt --workspace /path/to/workspace
```

## Settings Location

Configuration uses a **layered system** with three levels (in order of precedence):

### 1. Environment Variables (Global)

Loaded from `.env` file in the working directory:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional LLM settings
SMART_LLM=gpt-4o                    # Model for complex reasoning
FAST_LLM=gpt-4o-mini                # Model for simple tasks
EMBEDDING_MODEL=text-embedding-3-small

# Optional search providers (for web search component)
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

### 2. Workspace Settings (`{workspace}/.autogpt/autogpt.yaml`)

Workspace-wide permissions that apply to **all agents** in this workspace:

```yaml
allow:
  - read_file({workspace}/**)
  - write_to_file({workspace}/**)
  - list_folder({workspace}/**)
  - web_search(*)

deny:
  - read_file(**.env)
  - read_file(**.env.*)
  - read_file(**.key)
  - read_file(**.pem)
  - execute_shell(rm -rf:*)
  - execute_shell(sudo:*)
```

Auto-generated with sensible defaults if missing.

### 3. Agent Settings (`{workspace}/.autogpt/agents/{id}/permissions.yaml`)

Agent-specific permission overrides:

```yaml
allow:
  - execute_python(*)
  - web_search(*)

deny:
  - execute_shell(*)
```

## Permissions

The permission system uses **pattern matching** with a **first-match-wins** evaluation order.

### Permission Check Order

1. Agent deny list → **Block**
2. Workspace deny list → **Block**
3. Agent allow list → **Allow**
4. Workspace allow list → **Allow**
5. Session denied list → **Block** (commands denied during this session)
6. **Prompt user** → Interactive approval (if in interactive mode)

### Pattern Syntax

Format: `command_name(glob_pattern)`

| Pattern | Description |
|---------|-------------|
| `read_file({workspace}/**)` | Read any file in workspace (recursive) |
| `write_to_file({workspace}/*.txt)` | Write only .txt files in workspace root |
| `execute_shell(python:**)` | Execute Python commands only |
| `execute_shell(git:*)` | Execute any git command |
| `web_search(*)` | Allow all web searches |

Special tokens:
- `{workspace}` - Replaced with actual workspace path
- `**` - Matches any path including `/`
- `*` - Matches any characters except `/`

### Interactive Approval Scopes

When prompted for permission, users can choose:

| Scope | Effect |
|-------|--------|
| **Once** | Allow this one time only (not saved) |
| **Agent** | Always allow for this agent (saves to agent `permissions.yaml`) |
| **Workspace** | Always allow for all agents (saves to `autogpt.yaml`) |
| **Deny** | Deny this command (saves to appropriate deny list) |

### Default Security

Out of the box, the following are **denied by default**:
- Reading sensitive files (`.env`, `.key`, `.pem`)
- Destructive shell commands (`rm -rf`, `sudo`)
- Operations outside the workspace directory
