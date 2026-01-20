# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoGPT Classic is an experimental, **unsupported** project demonstrating autonomous GPT-4 operation. Dependencies will not be updated, and the codebase contains known vulnerabilities. This is preserved for educational/historical purposes.

## Repository Structure

```
/forge            - Core autonomous agent framework (main library)
/original_autogpt - Original AutoGPT implementation (depends on forge)
/direct_benchmark - Benchmark harness for testing agent performance
```

Each Python subproject has its own `pyproject.toml` and uses Poetry for dependency management.

## Common Commands

### Setup & Install
```bash
# Install forge (core library)
cd forge && poetry install

# Install original_autogpt (includes forge as dependency)
cd original_autogpt && poetry install

# Install benchmark
cd benchmark && poetry install

# Install with benchmark support (optional extra)
cd forge && poetry install --extras benchmark
cd original_autogpt && poetry install --extras benchmark
```

### Running Agents
```bash
# Run forge agent (from forge directory)
cd forge && poetry run python -m forge

# Run original autogpt (from original_autogpt directory)
cd original_autogpt && poetry run serve --debug

# Run autogpt CLI
cd original_autogpt && poetry run autogpt
```

Agents run on `http://localhost:8000` by default.

### Benchmarking
```bash
# Run benchmarks
cd direct_benchmark && poetry run python -m direct_benchmark run

# Run specific strategies and models
poetry run python -m direct_benchmark run \
    --strategies one_shot,rewoo \
    --models claude \
    --parallel 4

# Run a single test
poetry run python -m direct_benchmark run --tests ReadFile

# List available commands
poetry run python -m direct_benchmark --help
```

### Testing
```bash
cd forge && poetry run pytest                    # All tests
cd forge && poetry run pytest tests/             # Tests directory only
cd forge && poetry run pytest -k test_name       # Single test by name
cd forge && poetry run pytest path/to/test.py   # Specific test file
cd forge && poetry run pytest --cov             # With coverage
```

### Linting & Formatting

Run from forge/ or original_autogpt/ directory:

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
Depends on forge via local path (`autogpt-forge = { path = "../forge" }`):
- `autogpt/app/` - CLI application entry points
- `autogpt/agents/` - Agent implementations
- `autogpt/agent_factory/` - Agent creation logic

### Direct Benchmark
Benchmark harness for testing agent performance:
- `direct_benchmark/` - CLI and harness code
- `benchmark/agbenchmark/challenges/` - Test cases organized by category (code, retrieval, data, etc.)
- Reports generated in `direct_benchmark/reports/`

### Dependency Chain
`original_autogpt` → `forge`
`direct_benchmark` → `original_autogpt` → `forge`

## Code Style

- Python 3.12 target
- Line length: 88 characters (Black default)
- Black for formatting, isort for imports (profile="black")
- Type hints with Pyright checking

## Testing Patterns

- VCR cassettes in `/forge/tests/vcr_cassettes/` for HTTP mocking
- Async support via pytest-asyncio
- Fixtures defined in `conftest.py` files provide: `tmp_project_root`, `storage`, `config`, `llm_provider`, `agent`
- Tests require `OPENAI_API_KEY` environment variable (defaults to "sk-dummy" for mocked tests)

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
