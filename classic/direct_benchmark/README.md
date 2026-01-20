# Direct Benchmark Harness

High-performance benchmark harness for AutoGPT that directly instantiates agents without HTTP server overhead, enabling parallel execution of multiple configurations.

## Features

- **Direct Agent Instantiation**: No HTTP server, no Agent Protocol overhead
- **Parallel Execution**: Run multiple strategy/model combinations concurrently
- **Multiple Attempts**: Run each challenge multiple times for statistical reliability
- **Rich UI**: Live progress display with Rich library
- **Multiple Output Modes**: Default (rich), quiet, verbose, JSON for CI
- **Full CLI Compatibility**: All flags from the original agbenchmark supported

## Installation

All commands run from the `classic/` directory (parent of this directory):

```bash
cd classic
poetry install
```

## Usage

```bash
# Run benchmarks with default settings
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

# Run multiple attempts per challenge
poetry run direct-benchmark run \
    --strategies one_shot \
    --attempts 3

# Run only regression tests (previously beaten)
poetry run direct-benchmark run --maintain

# Run only non-regression tests (not consistently beaten)
poetry run direct-benchmark run --improve

# Run only never-beaten challenges
poetry run direct-benchmark run --explore

# List available challenges
poetry run direct-benchmark list-challenges

# List model presets
poetry run direct-benchmark list-models

# List strategies
poetry run direct-benchmark list-strategies
```

## CLI Options

### Challenge Selection
- `--strategies, -s`: Comma-separated strategies (one_shot, rewoo, plan_execute, reflexion, tree_of_thoughts)
- `--models, -m`: Comma-separated model presets (claude, openai, etc.)
- `--categories, -c`: Filter by challenge categories
- `--skip-category, -S`: Exclude categories
- `--tests, -t`: Filter by test names

### Execution Control
- `--attempts, -N`: Number of times to run each challenge
- `--parallel, -p`: Maximum parallel runs (default: 4)
- `--timeout`: Per-challenge timeout in seconds (default: 300)
- `--cutoff`: Alias for --timeout
- `--no-cutoff, --nc`: Disable time limit
- `--max-steps`: Maximum steps per challenge (default: 50)

### Challenge Filtering Modes
- `--maintain`: Run only regression tests (previously beaten consistently)
- `--improve`: Run only non-regression tests (not consistently beaten)
- `--explore`: Run only challenges that have never been beaten
- `--no-dep`: Run all challenges regardless of dependency success/failure

### Output & Debug
- `--quiet, -q`: Minimal output
- `--verbose, -v`: Detailed per-challenge output
- `--json`: JSON output for CI/scripting
- `--debug`: Enable debug output
- `--keep-answers`: Keep answer files for debugging

### Paths
- `--workspace`: Workspace root directory
- `--challenges-dir`: Path to challenges directory
- `--reports-dir`: Path to reports directory

## Available Strategies

| Strategy | Description |
|----------|-------------|
| `one_shot` | Single-pass reasoning (default, most reliable) |
| `rewoo` | Reasoning with observations |
| `plan_execute` | Plan then execute |
| `reflexion` | Self-reflection loop |
| `tree_of_thoughts` | Multiple reasoning paths |

## Available Model Presets

### Claude
- `claude`: sonnet-4 smart, haiku fast (default)
- `claude-smart`: sonnet-4 for both
- `claude-fast`: haiku for both
- `claude-opus`: opus smart, sonnet fast
- `claude-opus-only`: opus for both

### Claude with Extended Thinking
- `claude-thinking-10k`: 10k thinking tokens
- `claude-thinking-25k`: 25k thinking tokens
- `claude-thinking-50k`: 50k thinking tokens
- `claude-opus-thinking`: opus with 25k thinking
- `claude-opus-thinking-50k`: opus with 50k thinking

### OpenAI
- `openai`: gpt-4o smart, gpt-4o-mini fast
- `openai-smart`: gpt-4o for both
- `openai-fast`: gpt-4o-mini for both
- `gpt5`: gpt-5 smart, gpt-4o fast
- `gpt5-only`: gpt-5 for both

### OpenAI Reasoning Models
- `o1`, `o1-mini`: o1 variants
- `o1-low`, `o1-medium`, `o1-high`: o1 with reasoning effort
- `o3-low`, `o3-medium`, `o3-high`: o3 with reasoning effort

## Reports

Reports are generated in `./reports/` with format:
```
reports/
├── {timestamp}_{strategy}_{model}/
│   └── report.json
└── strategy_comparison_{timestamp}.json
```

## Key Differences from agbenchmark

| agbenchmark | direct_benchmark |
|-------------|------------------|
| `subprocess.Popen` + HTTP server | Direct `create_agent()` |
| HTTP/REST via Agent Protocol | Direct `propose_action()`/`execute()` |
| Sequential (one config at a time) | Parallel via asyncio semaphore |
| Port-based isolation | Workspace-based isolation |
