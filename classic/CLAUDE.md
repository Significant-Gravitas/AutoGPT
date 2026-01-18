# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoGPT Classic is an experimental, **unsupported** project demonstrating autonomous GPT-4 operation. Dependencies will not be updated, and the codebase contains known vulnerabilities. This is preserved for educational/historical purposes.

## Repository Structure

```
/forge            - Core autonomous agent framework (main library)
/original_autogpt - Original AutoGPT implementation (depends on forge)
/benchmark        - Performance testing/benchmarking tools
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
# Run benchmarks against an agent
cd benchmark && poetry run agbenchmark

# Or from forge/original_autogpt with benchmark extra installed
cd forge && poetry run agbenchmark
cd original_autogpt && poetry run agbenchmark
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
```bash
poetry run black .       # Format code
poetry run isort .       # Sort imports
poetry run flake8        # Lint
poetry run pyright       # Type check
```

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

### Benchmark
Independent testing framework for evaluating agent performance:
- `agbenchmark/challenges/` - Test cases organized by category (code, retrieval, memory, etc.)
- `agbenchmark/reports/` - Benchmark result reporting

### Dependency Chain
`original_autogpt` → `forge` ← `benchmark` (optional extra)

## Code Style

- Python 3.10 target
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
