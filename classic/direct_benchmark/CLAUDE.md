# CLAUDE.md - Direct Benchmark Harness

This file provides guidance to Claude Code when working with the direct benchmark harness.

## Overview

The Direct Benchmark Harness is a high-performance testing framework for AutoGPT that directly instantiates agents without HTTP server overhead. It enables parallel execution of multiple strategy/model configurations.

## Quick Reference

All commands run from the `classic/` directory (parent of this directory):

```bash
# Install (one-time setup)
cd classic
poetry install

# Run benchmarks
poetry run direct-benchmark run

# Run specific strategies and models
poetry run direct-benchmark run \
    --strategies one_shot,rewoo \
    --models claude,openai \
    --parallel 4

# Run a single test
poetry run direct-benchmark run \
    --strategies one_shot \
    --tests ReadFile

# List available challenges
poetry run direct-benchmark list-challenges

# List model presets
poetry run direct-benchmark list-models

# List strategies
poetry run direct-benchmark list-strategies
```

## CLI Options

### Run Command

| Option | Short | Description |
|--------|-------|-------------|
| `--strategies` | `-s` | Comma-separated strategies (one_shot, rewoo, plan_execute, reflexion, tree_of_thoughts) |
| `--models` | `-m` | Comma-separated model presets (claude, openai, etc.) |
| `--categories` | `-c` | Filter by challenge categories |
| `--skip-category` | `-S` | Exclude categories |
| `--tests` | `-t` | Filter by test names |
| `--attempts` | `-N` | Number of times to run each challenge |
| `--parallel` | `-p` | Maximum parallel runs (default: 4) |
| `--timeout` | | Per-challenge timeout in seconds (default: 300) |
| `--cutoff` | | Alias for --timeout |
| `--no-cutoff` | `--nc` | Disable time limit |
| `--max-steps` | | Maximum steps per challenge (default: 50) |
| `--maintain` | | Run only regression tests |
| `--improve` | | Run only non-regression tests |
| `--explore` | | Run only never-beaten challenges |
| `--no-dep` | | Ignore challenge dependencies |
| `--workspace` | | Workspace root directory |
| `--challenges-dir` | | Path to challenges directory |
| `--reports-dir` | | Path to reports directory |
| `--keep-answers` | | Keep answer files for debugging |
| `--quiet` | `-q` | Minimal output |
| `--verbose` | `-v` | Detailed per-challenge output |
| `--json` | | JSON output for CI/scripting |
| `--ci` | | CI mode: no live display, shows completion blocks (auto-enabled when CI env var is set or not a TTY) |
| `--fresh` | | Clear all saved state and start fresh (don't resume) |
| `--retry-failures` | | Re-run only the challenges that failed in previous run |
| `--reset-strategy` | | Reset saved results for specific strategy (can repeat) |
| `--reset-model` | | Reset saved results for specific model (can repeat) |
| `--reset-challenge` | | Reset saved results for specific challenge (can repeat) |
| `--debug` | | Enable debug output |

### State Management Commands
```bash
# Show current state
poetry run direct-benchmark state show

# Clear all state
poetry run direct-benchmark state clear

# Reset specific strategy/model/challenge
poetry run direct-benchmark state reset --strategy reflexion
poetry run direct-benchmark state reset --model claude-thinking-25k
poetry run direct-benchmark state reset --challenge ThreeSum
```

## Available Strategies

- `one_shot` - Single-pass reasoning (default)
- `rewoo` - Reasoning with observations
- `plan_execute` - Plan then execute
- `reflexion` - Self-reflection loop
- `tree_of_thoughts` - Multiple reasoning paths

## Available Model Presets

### Claude
- `claude` - sonnet-4 smart, haiku fast
- `claude-smart` - sonnet-4 for both
- `claude-fast` - haiku for both
- `claude-opus` - opus smart, sonnet fast
- `claude-opus-only` - opus for both

### Claude with Extended Thinking
- `claude-thinking-10k` - 10k thinking tokens
- `claude-thinking-25k` - 25k thinking tokens
- `claude-thinking-50k` - 50k thinking tokens
- `claude-opus-thinking` - opus with 25k thinking
- `claude-opus-thinking-50k` - opus with 50k thinking

### OpenAI
- `openai` - gpt-4o smart, gpt-4o-mini fast
- `openai-smart` - gpt-4o for both
- `openai-fast` - gpt-4o-mini for both
- `gpt5` - gpt-5 smart, gpt-4o fast
- `gpt5-only` - gpt-5 for both

### OpenAI Reasoning Models
- `o1`, `o1-mini` - o1 variants
- `o1-low`, `o1-medium`, `o1-high` - o1 with reasoning effort
- `o3-low`, `o3-medium`, `o3-high` - o3 with reasoning effort
- `gpt5-low`, `gpt5-medium`, `gpt5-high` - gpt-5 with reasoning effort

## Directory Structure

```
direct_benchmark/
├── pyproject.toml           # Poetry config
├── README.md                 # User documentation
├── CLAUDE.md                 # This file
├── .gitignore
└── direct_benchmark/
    ├── __init__.py
    ├── __main__.py           # CLI entry point
    ├── models.py             # Pydantic models, presets
    ├── harness.py            # Main orchestrator
    ├── runner.py             # AgentRunner (single agent lifecycle)
    ├── parallel.py           # ParallelExecutor (concurrent runs)
    ├── challenge_loader.py   # Load challenges from JSON
    ├── evaluator.py          # Evaluate outputs vs ground truth
    ├── report.py             # Report generation
    └── ui.py                 # Rich UI components
```

## Architecture

### Execution Flow

```
CLI args → HarnessConfig
    ↓
BenchmarkHarness.run()
    ↓
ChallengeLoader.load_all() → list[Challenge]
    ↓
ParallelExecutor.execute_matrix(configs × challenges × attempts)
    ↓
[Parallel with semaphore limiting to N concurrent]
    ↓
AgentRunner.run_challenge():
  1. Create temp workspace
  2. Copy input artifacts to agent workspace
  3. Create AppConfig with strategy/model
  4. create_agent() - direct instantiation
  5. Run agent loop until finish/timeout
  6. Collect output files
    ↓
Evaluator.evaluate() - check against ground truth
    ↓
ReportGenerator - write reports
```

### Key Components

**AgentRunner** (`runner.py`)
- Manages single agent lifecycle for one challenge
- Creates isolated temp workspace per run
- Copies input artifacts to `{workspace}/.autogpt/agents/{agent_id}/workspace/`
- Instantiates agent directly via `create_agent()`
- Runs agent loop: `propose_action()` → `execute()` until finish/timeout

**ParallelExecutor** (`parallel.py`)
- Manages concurrent execution with asyncio semaphore
- Supports multiple attempts per challenge
- Reports progress via callbacks

**Evaluator** (`evaluator.py`)
- String matching (should_contain/should_not_contain)
- Python script execution
- Pytest execution

**ReportGenerator** (`report.py`)
- Per-config `report.json` files (compatible with agbenchmark format)
- Comparison reports across all configs

## Report Format

Reports are generated in `./reports/` with format:
```
reports/
├── {timestamp}_{strategy}_{model}/
│   └── report.json
└── strategy_comparison_{timestamp}.json
```

## Dependencies

- `autogpt-forge` - Core agent framework
- `autogpt` - Original AutoGPT agent
- `click` - CLI framework
- `pydantic` - Data models
- `rich` - Terminal UI

## Key Differences from agbenchmark

| agbenchmark | direct_benchmark |
|-------------|-----------------|
| `subprocess.Popen` + HTTP server | Direct `create_agent()` |
| HTTP/REST via Agent Protocol | Direct `propose_action()`/`execute()` |
| Sequential (one config at a time) | Parallel via asyncio semaphore |
| Port-based isolation | Workspace-based isolation |
| `agbenchmark run` CLI | Direct JSON parsing |

## Common Tasks

### Run Full Benchmark Suite
```bash
poetry run direct-benchmark run \
    --strategies one_shot,rewoo,plan_execute \
    --models claude \
    --parallel 8
```

### Compare Strategies
```bash
poetry run direct-benchmark run \
    --strategies one_shot,rewoo,plan_execute,reflexion \
    --models claude \
    --tests ReadFile,WriteFile,ThreeSum
```

### Debug a Failing Test
```bash
poetry run direct-benchmark run \
    --strategies one_shot \
    --tests FailingTest \
    --keep-answers \
    --verbose
```

### Resume / Incremental Runs
The benchmark automatically saves progress and resumes from where it left off.
State is saved to `.benchmark_state.json` in the reports directory.

```bash
# Run benchmarks - will resume from last run automatically
poetry run direct-benchmark run \
    --strategies one_shot,reflexion \
    --models claude

# Start fresh (clear all saved state)
poetry run direct-benchmark run --fresh \
    --strategies one_shot,reflexion \
    --models claude

# Reset specific strategy and re-run
poetry run direct-benchmark run \
    --reset-strategy reflexion \
    --strategies one_shot,reflexion \
    --models claude

# Reset specific model and re-run
poetry run direct-benchmark run \
    --reset-model claude-thinking-25k \
    --strategies one_shot \
    --models claude,claude-thinking-25k

# Retry only the failures from the last run
poetry run direct-benchmark run --retry-failures \
    --strategies one_shot,reflexion \
    --models claude
```

### CI/Scripting Mode
```bash
# JSON output (parseable)
poetry run direct-benchmark run --json

# CI mode - shows completion blocks without Live display
# Auto-enabled when CI=true env var is set or stdout is not a TTY
poetry run direct-benchmark run --ci
```
