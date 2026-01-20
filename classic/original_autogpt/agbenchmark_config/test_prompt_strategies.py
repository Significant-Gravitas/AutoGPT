#!/usr/bin/env python3
"""
Unified Prompt Strategy Benchmark Suite

The main entry point for benchmarking AutoGPT with different prompt strategies
and LLM configurations. Runs benchmarks and automatically analyzes results.

USAGE:
    # Run full benchmark suite (all strategies, all tests)
    poetry run python agbenchmark_config/test_prompt_strategies.py

    # Quick smoke test
    poetry run python agbenchmark_config/test_prompt_strategies.py --quick

    # Run specific strategies
    poetry run python agbenchmark_config/test_prompt_strategies.py --strategies one_shot,rewoo

    # Run specific test categories
    poetry run python agbenchmark_config/test_prompt_strategies.py --categories code,retrieval

    # Compare existing reports (no new tests)
    poetry run python agbenchmark_config/test_prompt_strategies.py --compare-only

    # Analyze failures only (no benchmarks)
    poetry run python agbenchmark_config/test_prompt_strategies.py --analyze-only

    # Run with specific model configurations
    poetry run python agbenchmark_config/test_prompt_strategies.py --models claude,openai

    # Skip failure analysis after benchmarks
    poetry run python agbenchmark_config/test_prompt_strategies.py --no-analyze

    # Compare models: Claude vs OpenAI on one_shot
    poetry run python agbenchmark_config/test_prompt_strategies.py --models claude,openai --strategies one_shot
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Available prompt strategies
STRATEGIES = [
    "one_shot",
    "rewoo",
    "plan_execute",
    "reflexion",
    "tree_of_thoughts",
]

# Quick test categories for smoke testing
# Valid categories: general, ethereum, data, scrape_synthesize, coding
QUICK_TEST_CATEGORIES = ["general"]

# Default timeout for agent startup (seconds)
AGENT_STARTUP_TIMEOUT = 30

# Default timeout for benchmark run (seconds)
BENCHMARK_TIMEOUT = 3600  # 1 hour


@dataclass
class ModelConfig:
    """Configuration for LLM models."""

    name: str  # Display name for the configuration
    smart_llm: str  # Model for complex reasoning tasks
    fast_llm: str  # Model for quick operations
    thinking_budget_tokens: Optional[int] = None  # Extended thinking budget (Anthropic)
    reasoning_effort: Optional[str] = None  # Reasoning effort (OpenAI o-series/GPT-5)

    def to_env(self) -> dict[str, str]:
        """Return environment variables for this config."""
        env: dict[str, str] = {}
        if self.smart_llm:
            env["SMART_LLM"] = self.smart_llm
        if self.fast_llm:
            env["FAST_LLM"] = self.fast_llm
        if self.thinking_budget_tokens:
            env["THINKING_BUDGET_TOKENS"] = str(self.thinking_budget_tokens)
        if self.reasoning_effort:
            env["REASONING_EFFORT"] = self.reasoning_effort
        return env

    def __str__(self) -> str:
        parts = [f"smart={self.smart_llm}", f"fast={self.fast_llm}"]
        if self.thinking_budget_tokens:
            parts.append(f"thinking={self.thinking_budget_tokens}")
        if self.reasoning_effort:
            parts.append(f"reasoning={self.reasoning_effort}")
        return f"{self.name} ({', '.join(parts)})"


# Preset model configurations
MODEL_PRESETS: dict[str, ModelConfig] = {
    # Claude configurations
    "claude": ModelConfig(
        name="claude",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-3-5-haiku-20241022",
    ),
    "claude-smart": ModelConfig(
        name="claude-smart",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-sonnet-4-20250514",
    ),
    "claude-fast": ModelConfig(
        name="claude-fast",
        smart_llm="claude-3-5-haiku-20241022",
        fast_llm="claude-3-5-haiku-20241022",
    ),
    "claude-opus": ModelConfig(
        name="claude-opus",
        smart_llm="claude-opus-4-5-20251101",
        fast_llm="claude-sonnet-4-20250514",
    ),
    "claude-opus-only": ModelConfig(
        name="claude-opus-only",
        smart_llm="claude-opus-4-5-20251101",
        fast_llm="claude-opus-4-5-20251101",
    ),
    # OpenAI configurations
    "openai": ModelConfig(
        name="openai",
        smart_llm="gpt-4o",
        fast_llm="gpt-4o-mini",
    ),
    "openai-smart": ModelConfig(
        name="openai-smart",
        smart_llm="gpt-4o",
        fast_llm="gpt-4o",
    ),
    "openai-fast": ModelConfig(
        name="openai-fast",
        smart_llm="gpt-4o-mini",
        fast_llm="gpt-4o-mini",
    ),
    "gpt5": ModelConfig(
        name="gpt5",
        smart_llm="gpt-5",
        fast_llm="gpt-4o",
    ),
    "gpt5-only": ModelConfig(
        name="gpt5-only",
        smart_llm="gpt-5",
        fast_llm="gpt-5",
    ),
    "o1": ModelConfig(
        name="o1",
        smart_llm="o1",
        fast_llm="gpt-4o-mini",
    ),
    "o1-mini": ModelConfig(
        name="o1-mini",
        smart_llm="o1-mini",
        fast_llm="gpt-4o-mini",
    ),
    # Claude extended thinking configurations
    "claude-thinking-10k": ModelConfig(
        name="claude-thinking-10k",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-3-5-haiku-20241022",
        thinking_budget_tokens=10000,
    ),
    "claude-thinking-25k": ModelConfig(
        name="claude-thinking-25k",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-3-5-haiku-20241022",
        thinking_budget_tokens=25000,
    ),
    "claude-thinking-50k": ModelConfig(
        name="claude-thinking-50k",
        smart_llm="claude-sonnet-4-20250514",
        fast_llm="claude-3-5-haiku-20241022",
        thinking_budget_tokens=50000,
    ),
    "claude-opus-thinking": ModelConfig(
        name="claude-opus-thinking",
        smart_llm="claude-opus-4-5-20251101",
        fast_llm="claude-sonnet-4-20250514",
        thinking_budget_tokens=25000,
    ),
    "claude-opus-thinking-50k": ModelConfig(
        name="claude-opus-thinking-50k",
        smart_llm="claude-opus-4-5-20251101",
        fast_llm="claude-sonnet-4-20250514",
        thinking_budget_tokens=50000,
    ),
    # OpenAI reasoning effort configurations
    "o1-low": ModelConfig(
        name="o1-low",
        smart_llm="o1",
        fast_llm="gpt-4o-mini",
        reasoning_effort="low",
    ),
    "o1-medium": ModelConfig(
        name="o1-medium",
        smart_llm="o1",
        fast_llm="gpt-4o-mini",
        reasoning_effort="medium",
    ),
    "o1-high": ModelConfig(
        name="o1-high",
        smart_llm="o1",
        fast_llm="gpt-4o-mini",
        reasoning_effort="high",
    ),
    "o3-low": ModelConfig(
        name="o3-low",
        smart_llm="o3",
        fast_llm="gpt-4o-mini",
        reasoning_effort="low",
    ),
    "o3-medium": ModelConfig(
        name="o3-medium",
        smart_llm="o3",
        fast_llm="gpt-4o-mini",
        reasoning_effort="medium",
    ),
    "o3-high": ModelConfig(
        name="o3-high",
        smart_llm="o3",
        fast_llm="gpt-4o-mini",
        reasoning_effort="high",
    ),
    "gpt5-low": ModelConfig(
        name="gpt5-low",
        smart_llm="gpt-5",
        fast_llm="gpt-4o",
        reasoning_effort="low",
    ),
    "gpt5-medium": ModelConfig(
        name="gpt5-medium",
        smart_llm="gpt-5",
        fast_llm="gpt-4o",
        reasoning_effort="medium",
    ),
    "gpt5-high": ModelConfig(
        name="gpt5-high",
        smart_llm="gpt-5",
        fast_llm="gpt-4o",
        reasoning_effort="high",
    ),
}

# Default model config (uses environment defaults)
DEFAULT_MODEL_CONFIG = ModelConfig(
    name="default",
    smart_llm="",  # Empty means use env default
    fast_llm="",
)

# Matrix mode presets - curated set for comprehensive comparison
# These are grouped by purpose for the --all flag
MATRIX_MODELS = {
    # Core models (always run in --all)
    "core": [
        "claude",  # Claude Sonnet + Haiku
        "openai",  # GPT-4o + GPT-4o-mini
    ],
    # Extended thinking variants
    "thinking": [
        "claude-thinking-10k",
        "claude-thinking-25k",
        "claude-opus-thinking",
    ],
    # Reasoning effort variants (OpenAI o-series)
    "reasoning": [
        "o1-medium",
        "o1-high",
    ],
    # Premium models (expensive but powerful)
    "premium": [
        "claude-opus",
        "gpt5",
    ],
}


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run (strategy + model combination)."""

    strategy: str
    model_config: ModelConfig
    report_dir: Path
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    total_time: float = 0.0
    total_cost: float = 0.0
    avg_steps: float = 0.0
    test_results: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.tests_run == 0:
            return 0.0
        return self.tests_passed / self.tests_run * 100

    @property
    def config_name(self) -> str:
        """Return a unique name for this strategy+model combination."""
        if self.model_config.name == "default":
            return self.strategy
        return f"{self.strategy}/{self.model_config.name}"


# Alias for backwards compatibility
StrategyResult = BenchmarkResult


@dataclass
class ComparisonReport:
    """Comparison report across all configurations."""

    timestamp: str
    configurations: list[str]  # List of config names (strategy/model combinations)
    results: dict[str, BenchmarkResult]
    test_names: list[str]
    # Keep strategies for backwards compatibility
    strategies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "configurations": self.configurations,
            "strategies": self.strategies,
            "results": {
                name: {
                    "strategy": r.strategy,
                    "model_config": {
                        "name": r.model_config.name,
                        "smart_llm": r.model_config.smart_llm,
                        "fast_llm": r.model_config.fast_llm,
                    },
                    "report_dir": str(r.report_dir),
                    "tests_run": r.tests_run,
                    "tests_passed": r.tests_passed,
                    "tests_failed": r.tests_failed,
                    "success_rate": r.success_rate,
                    "total_time": r.total_time,
                    "total_cost": r.total_cost,
                    "avg_steps": r.avg_steps,
                    "test_results": r.test_results,
                }
                for name, r in self.results.items()
            },
            "test_names": self.test_names,
        }


def find_python() -> str:
    """Find the poetry-managed Python."""
    return "poetry"


def log(msg: str, level: str = "INFO") -> None:
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)


def log_progress(msg: str) -> None:
    """Print a progress message without newline."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", end="", flush=True)


def start_agent(
    strategy: str,
    model_config: ModelConfig,
    port: int = 8000,
    show_agent_output: bool = False,
) -> subprocess.Popen:
    """Start the AutoGPT agent with a specific strategy and model config."""
    env = os.environ.copy()
    env["PROMPT_STRATEGY"] = strategy
    env["AP_SERVER_PORT"] = str(port)
    env["NONINTERACTIVE_MODE"] = "True"

    # Set model configuration if specified
    model_env = model_config.to_env()
    for key, value in model_env.items():
        if value:  # Only set if not empty
            env[key] = value

    model_desc = f" with {model_config.name}" if model_config.name != "default" else ""
    log(f"Starting agent with strategy '{strategy}'{model_desc} on port {port}...")
    log(f"  PROMPT_STRATEGY:      {env['PROMPT_STRATEGY']}")
    log(f"  NONINTERACTIVE_MODE:  {env.get('NONINTERACTIVE_MODE', 'not set')}")
    log(f"  SMART_LLM:            {env.get('SMART_LLM', '(env default)')}")
    log(f"  FAST_LLM:             {env.get('FAST_LLM', '(env default)')}")
    if model_config.thinking_budget_tokens:
        log(f"  THINKING_BUDGET:      {model_config.thinking_budget_tokens} tokens")
    if model_config.reasoning_effort:
        log(f"  REASONING_EFFORT:     {model_config.reasoning_effort}")

    # Start the agent server (port is set via AP_SERVER_PORT env var)
    proc = subprocess.Popen(
        ["poetry", "run", "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent.parent,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Wait for agent to be ready, streaming output
    import select
    import threading

    # Thread to read and print agent output
    def stream_output():
        if proc.stdout:
            for line in proc.stdout:
                if show_agent_output:
                    print(f"    [agent] {line.rstrip()}", flush=True)

    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()

    log("  Waiting for agent to be ready...")
    start_time = time.time()
    while time.time() - start_time < AGENT_STARTUP_TIMEOUT:
        try:
            import urllib.request

            urllib.request.urlopen(
                f"http://localhost:{port}/ap/v1/agent/tasks", timeout=2
            )
            elapsed = time.time() - start_time
            log(f"Agent ready on port {port} (took {elapsed:.1f}s)")
            return proc
        except Exception:
            time.sleep(0.5)

    proc.kill()
    raise TimeoutError(f"Agent failed to start within {AGENT_STARTUP_TIMEOUT}s")


def stop_agent(proc: subprocess.Popen) -> None:
    """Stop the agent process."""
    log("Stopping agent...")
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log("Agent didn't stop gracefully, killing...", "WARN")
            proc.kill()
            proc.wait()
    log("Agent stopped")


def run_benchmark(
    strategy: str,
    port: int = 8000,
    categories: Optional[list[str]] = None,
    tests: Optional[list[str]] = None,
    attempts: int = 1,
    timeout: int = BENCHMARK_TIMEOUT,
    verbose: bool = True,
) -> Optional[Path]:
    """Run the agbenchmark and return the report directory."""
    cmd = ["poetry", "run", "agbenchmark", "run"]

    if attempts > 1:
        cmd.extend(["--attempts", str(attempts)])

    if categories:
        for cat in categories:
            cmd.extend(["--category", cat])

    if tests:
        for test in tests:
            cmd.extend(["--test", test])

    # Set the host to the correct port
    env = os.environ.copy()
    env["AGENT_API_URL"] = f"http://localhost:{port}"
    # Force unbuffered output for real-time logging
    env["PYTHONUNBUFFERED"] = "1"

    log(f"Running benchmark: {' '.join(cmd)}")
    log(f"Timeout: {timeout}s ({timeout // 60} minutes)")
    benchmark_start = time.time()

    try:
        # Use Popen to stream output in real-time
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Stream output in real-time
        last_output_time = time.time()
        output_lines = []

        while True:
            # Check if process has finished
            retcode = proc.poll()

            # Try to read a line (non-blocking would be ideal but this works)
            if proc.stdout:
                line = proc.stdout.readline()
                if line:
                    last_output_time = time.time()
                    output_lines.append(line)
                    if verbose:
                        # Indent benchmark output for clarity
                        print(f"    | {line.rstrip()}", flush=True)

            # Check for timeout
            elapsed = time.time() - benchmark_start
            if elapsed > timeout:
                log(f"Benchmark timed out after {elapsed:.0f}s", "ERROR")
                proc.kill()
                proc.wait()
                return None

            # Warn if no output for a while
            silence_duration = time.time() - last_output_time
            if silence_duration > 60 and int(silence_duration) % 60 == 0:
                log(
                    f"No output for {int(silence_duration)}s "
                    f"(elapsed: {int(elapsed)}s)...",
                    "WARN",
                )

            # Process finished
            if retcode is not None:
                # Read any remaining output
                if proc.stdout:
                    for line in proc.stdout:
                        output_lines.append(line)
                        if verbose:
                            print(f"    | {line.rstrip()}", flush=True)
                break

            time.sleep(0.1)

        elapsed = time.time() - benchmark_start
        log(f"Benchmark completed with code {retcode} (took {elapsed:.1f}s)")

    except Exception as e:
        log(f"Benchmark failed with exception: {e}", "ERROR")
        return None

    # Find the most recent report directory
    reports_dir = Path(__file__).parent / "reports"
    if not reports_dir.exists():
        log("No reports directory found", "WARN")
        return None

    report_dirs = sorted(
        [
            d
            for d in reports_dir.iterdir()
            if d.is_dir() and re.match(r"^\d{8}T\d{6}_", d.name)
        ],
        key=lambda d: d.name,
        reverse=True,
    )

    if report_dirs:
        log(f"Found report: {report_dirs[0].name}")
        return report_dirs[0]

    log("No report directory found after benchmark", "WARN")
    return None


def parse_report(
    report_dir: Path,
    strategy: str,
    model_config: Optional[ModelConfig] = None,
) -> BenchmarkResult:
    """Parse a benchmark report and extract metrics."""
    if model_config is None:
        model_config = DEFAULT_MODEL_CONFIG
    result = BenchmarkResult(
        strategy=strategy, model_config=model_config, report_dir=report_dir
    )

    report_file = report_dir / "report.json"
    if not report_file.exists():
        return result

    with open(report_file) as f:
        data = json.load(f)

    # Handle both in-progress and completed reports
    test_tree = data.get("tests", data)

    def process_tests(tests: dict, prefix: str = "") -> None:
        for test_name, test_data in tests.items():
            full_name = f"{prefix}{test_name}" if prefix else test_name

            if "tests" in test_data:
                # This is a test suite, recurse
                process_tests(test_data["tests"], f"{full_name}/")
                continue

            metrics = test_data.get("metrics", {})
            if not metrics.get("attempted", False):
                continue

            result.tests_run += 1

            # Get detailed results - success is in results[].success, not metrics
            test_results = test_data.get("results", [])
            if test_results:
                first_result = test_results[0]
                # Check success from the actual test result
                success = first_result.get("success", False)
                run_time_str = first_result.get("run_time", "0")
                # Handle formats like "5.698s", "5.698 second", "5.698 seconds"
                run_time_str = run_time_str.split()[0].rstrip("s")
                result.total_time += float(run_time_str)
                result.total_cost += first_result.get("cost", 0) or 0
                n_steps = first_result.get("n_steps", 0)
            else:
                # Fallback: check success_percentage in metrics
                success_pct = metrics.get("success_percentage", 0) or metrics.get(
                    "success_%", 0
                )
                success = success_pct is not None and success_pct > 0
                n_steps = 0

            if success:
                result.tests_passed += 1
            else:
                result.tests_failed += 1

            result.test_results[full_name] = {
                "success": success,
                "n_steps": n_steps,
                "difficulty": test_data.get("difficulty", "unknown"),
            }

    process_tests(test_tree)

    if result.tests_run > 0:
        total_steps = sum(t.get("n_steps") or 0 for t in result.test_results.values())
        result.avg_steps = total_steps / result.tests_run

    return result


def find_strategy_reports() -> dict[str, list[Path]]:
    """Find existing reports and try to identify which strategy they used."""
    reports_dir = Path(__file__).parent / "reports"
    if not reports_dir.exists():
        return {}

    # Group reports by apparent strategy (from directory naming or metadata)
    strategy_reports = defaultdict(list)

    for report_dir in reports_dir.iterdir():
        if not report_dir.is_dir():
            continue
        if not re.match(r"^\d{8}T\d{6}_", report_dir.name):
            continue

        report_file = report_dir / "report.json"
        if not report_file.exists():
            continue

        # Try to determine strategy from report metadata or naming
        with open(report_file) as f:
            data = json.load(f)

        # Check for strategy in metadata (if we added it)
        strategy = data.get("metadata", {}).get("prompt_strategy", "unknown")

        # Or check directory name suffix
        if strategy == "unknown":
            for s in STRATEGIES:
                if s in report_dir.name:
                    strategy = s
                    break

        strategy_reports[strategy].append(report_dir)

    return dict(strategy_reports)


def print_comparison_table(report: ComparisonReport) -> None:
    """Print a comparison table of results."""
    print("\n" + "=" * 80)
    print("PROMPT STRATEGY & MODEL COMPARISON REPORT")
    print("=" * 80)
    print(f"Timestamp: {report.timestamp}")
    print()

    # Use configurations if available, otherwise fall back to strategies
    config_list = report.configurations if report.configurations else report.strategies

    # Summary table
    print("SUMMARY")
    print("-" * 80)
    headers = [
        "Configuration",
        "Tests",
        "Passed",
        "Failed",
        "Success %",
        "Avg Steps",
        "Cost",
    ]
    rows = []

    for config_name in config_list:
        r = report.results.get(config_name)
        if r:
            rows.append(
                [
                    config_name,
                    r.tests_run,
                    r.tests_passed,
                    r.tests_failed,
                    f"{r.success_rate:.1f}%",
                    f"{r.avg_steps:.1f}",
                    f"${r.total_cost:.4f}",
                ]
            )
        else:
            rows.append([config_name, "-", "-", "-", "-", "-", "-"])

    # Print table
    col_widths = [
        max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))
    ]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)

    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))

    print()

    # Per-test comparison
    if report.test_names:
        print("PER-TEST RESULTS")
        print("-" * 80)

        test_headers = ["Test"] + list(config_list)
        test_rows = []

        for test_name in sorted(report.test_names):
            row = [test_name[:40]]  # Truncate long names
            for config_name in config_list:
                r = report.results.get(config_name)
                if r and test_name in r.test_results:
                    tr = r.test_results[test_name]
                    status = "✅" if tr["success"] else "❌"
                    n_steps = tr.get("n_steps") or 0
                    row.append(f"{status} ({n_steps} steps)")
                else:
                    row.append("-")
            test_rows.append(row)

        test_col_widths = [
            max(len(str(row[i])) for row in [test_headers] + test_rows)
            for i in range(len(test_headers))
        ]
        test_fmt = " | ".join(f"{{:<{w}}}" for w in test_col_widths)

        print(test_fmt.format(*test_headers))
        print("-+-".join("-" * w for w in test_col_widths))
        for row in test_rows:
            print(test_fmt.format(*row))

    print()
    print("=" * 80)


def run_benchmark_config(
    strategy: str,
    model_config: ModelConfig,
    port: int,
    categories: Optional[list[str]],
    tests: Optional[list[str]],
    attempts: int,
    verbose: bool = True,
    show_agent_output: bool = False,
) -> Optional[BenchmarkResult]:
    """Run benchmark for a single strategy and model configuration."""
    config_name = (
        f"{strategy}/{model_config.name}"
        if model_config.name != "default"
        else strategy
    )
    print(f"\n{'='*70}", flush=True)
    log(f"STARTING CONFIGURATION: {config_name}")
    print("=" * 70, flush=True)

    config_start = time.time()
    agent_proc = None
    try:
        # Start agent
        agent_proc = start_agent(strategy, model_config, port, show_agent_output)

        # Run benchmark
        report_dir = run_benchmark(
            strategy=strategy,
            port=port,
            categories=categories,
            tests=tests,
            attempts=attempts,
            verbose=verbose,
        )

        if report_dir:
            # Rename report directory to include config name
            safe_name = config_name.replace("/", "_")
            new_name = f"{report_dir.name}_{safe_name}"
            new_path = report_dir.parent / new_name
            if not new_path.exists():
                report_dir.rename(new_path)
                report_dir = new_path

            # Parse results
            result = parse_report(report_dir, strategy, model_config)
            elapsed = time.time() - config_start
            log(
                f"FINISHED {config_name}: "
                f"{result.tests_passed}/{result.tests_run} passed "
                f"(total time: {elapsed:.1f}s)"
            )
            return result
        else:
            log("No report generated", "WARN")
            return None

    except Exception as e:
        import traceback

        log(f"Error: {e}", "ERROR")
        traceback.print_exc()
        return None

    finally:
        if agent_proc:
            stop_agent(agent_proc)


# Alias for backwards compatibility
def run_strategy_benchmark(
    strategy: str,
    port: int,
    categories: Optional[list[str]],
    tests: Optional[list[str]],
    attempts: int,
) -> Optional[BenchmarkResult]:
    """Run benchmark for a single strategy (backwards compatible)."""
    return run_benchmark_config(
        strategy=strategy,
        model_config=DEFAULT_MODEL_CONFIG,
        port=port,
        categories=categories,
        tests=tests,
        attempts=attempts,
    )


def run_failure_analysis(
    reports_dir: Optional[Path] = None,
    strategy: Optional[str] = None,
    test_name: Optional[str] = None,
    export_markdown: bool = True,
) -> None:
    """Run failure analysis on benchmark reports.

    Args:
        reports_dir: Path to reports directory (default: ./reports)
        strategy: Focus on a specific strategy (optional)
        test_name: Compare a specific test across strategies (optional)
        export_markdown: Whether to export a markdown report (default: True)
    """
    if reports_dir is None:
        reports_dir = Path(__file__).parent / "reports"

    if not reports_dir.exists():
        log("No reports directory found, skipping failure analysis", "WARN")
        return

    print()
    print("=" * 70, flush=True)
    log("FAILURE ANALYSIS")
    print("=" * 70, flush=True)

    try:
        # Import the analyzer
        from analyze_failures import FailureAnalyzer
    except ImportError:
        # Try relative import
        try:
            import sys

            sys.path.insert(0, str(Path(__file__).parent))
            from analyze_failures import FailureAnalyzer
        except ImportError as e:
            log(f"Could not import failure analyzer: {e}", "WARN")
            return

    try:
        analyzer = FailureAnalyzer(reports_dir, use_llm=False)
        analyzer.analyze_all()

        if not analyzer.strategies:
            log("No strategy reports found for analysis", "WARN")
            return

        # Print results
        analyzer.print_summary()
        analyzer.print_pattern_analysis()

        if test_name:
            analyzer.compare_test(test_name)
        elif strategy:
            analyzer.print_failed_tests(strategy)
        else:
            analyzer.print_failed_tests()

        # Export markdown report
        if export_markdown:
            md_path = (
                reports_dir / f"failure_analysis_{datetime.now():%Y%m%dT%H%M%S}.md"
            )
            analyzer.export_markdown(md_path)

    except Exception as e:
        log(f"Failure analysis error: {e}", "ERROR")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Test harness for comparing prompt strategies and LLM models"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        help=f"Comma-separated list of strategies to test (default: all). "
        f"Available: {', '.join(STRATEGIES)}",
    )
    parser.add_argument(
        "--models",
        type=str,
        help=f"Comma-separated list of model presets to test. "
        f"Available: {', '.join(MODEL_PRESETS.keys())}",
    )
    parser.add_argument(
        "--smart-llm",
        type=str,
        help="Custom smart LLM model name (e.g., gpt-4o, claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--fast-llm",
        type=str,
        help="Custom fast LLM model name (e.g., gpt-4o-mini, claude-3-5-haiku)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        help="Override thinking budget tokens for Anthropic models (min 1024)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Override reasoning effort for OpenAI o-series/GPT-5 models",
    )
    parser.add_argument(
        "--categories",
        type=str,
        help="Comma-separated list of test categories to run",
    )
    parser.add_argument(
        "--tests",
        type=str,
        help="Comma-separated list of specific tests to run",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of attempts per test (default: 1)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test mode (basic tests only)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Full matrix mode: run all strategies across all core models",
    )
    parser.add_argument(
        "--matrix",
        type=str,
        help="Matrix groups to include (comma-separated). "
        f"Available: {', '.join(MATRIX_MODELS.keys())}. "
        "Use --all for core, or --matrix core,thinking,premium for custom",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing reports, don't run new tests",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run failure analysis on existing reports (no benchmarks)",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip failure analysis after benchmarks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run agent on (default: 8000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for comparison report JSON",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available model presets and exit",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress benchmark output (only show summary)",
    )
    parser.add_argument(
        "--agent-output",
        action="store_true",
        help="Show agent server output (useful for debugging config)",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Handle analyze-only mode first
    if args.analyze_only:
        log("Running failure analysis only...")
        run_failure_analysis()
        sys.exit(0)

    # List models if requested
    if args.list_models:
        print("Available model presets:")
        print("-" * 70)
        for name, config in MODEL_PRESETS.items():
            print(f"  {name}:")
            print(f"    smart_llm: {config.smart_llm}")
            print(f"    fast_llm:  {config.fast_llm}")
            if config.thinking_budget_tokens:
                print(f"    thinking_budget: {config.thinking_budget_tokens}")
            if config.reasoning_effort:
                print(f"    reasoning_effort: {config.reasoning_effort}")
        sys.exit(0)

    # Parse strategies
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
        invalid = [s for s in strategies if s not in STRATEGIES]
        if invalid:
            print(f"Invalid strategies: {invalid}")
            print(f"Available: {STRATEGIES}")
            sys.exit(1)
    else:
        strategies = STRATEGIES

    # Parse model configurations
    model_configs: list[ModelConfig] = []

    if args.all or args.matrix:
        # Matrix mode: build list from matrix groups
        if args.all:
            # --all uses core models
            matrix_groups = ["core"]
        else:
            # Parse custom matrix groups
            matrix_groups = [g.strip() for g in args.matrix.split(",")]
            invalid_groups = [g for g in matrix_groups if g not in MATRIX_MODELS]
            if invalid_groups:
                print(f"Invalid matrix groups: {invalid_groups}")
                print(f"Available: {list(MATRIX_MODELS.keys())}")
                sys.exit(1)

        # Collect all model names from selected groups
        model_names = []
        for group in matrix_groups:
            model_names.extend(MATRIX_MODELS[group])

        # Convert to ModelConfig objects
        model_configs = [MODEL_PRESETS[m] for m in model_names]
        log(
            f"Matrix mode: {len(model_configs)} model configurations from groups {matrix_groups}"
        )

    elif (
        args.smart_llm or args.fast_llm or args.thinking_budget or args.reasoning_effort
    ):
        # Custom model configuration
        custom_config = ModelConfig(
            name="custom",
            smart_llm=args.smart_llm or "",
            fast_llm=args.fast_llm or "",
            thinking_budget_tokens=args.thinking_budget,
            reasoning_effort=args.reasoning_effort,
        )
        model_configs.append(custom_config)
    elif args.models:
        # Parse model presets
        model_names = [m.strip() for m in args.models.split(",")]
        invalid_models = [m for m in model_names if m not in MODEL_PRESETS]
        if invalid_models:
            print(f"Invalid model presets: {invalid_models}")
            print(f"Available: {list(MODEL_PRESETS.keys())}")
            print("Use --list-models to see all presets with their configurations")
            sys.exit(1)
        model_configs = [MODEL_PRESETS[m] for m in model_names]
    else:
        # Default: use environment defaults (no model override)
        model_configs = [DEFAULT_MODEL_CONFIG]

    # Parse categories/tests
    categories = (
        [c.strip() for c in args.categories.split(",")] if args.categories else None
    )
    tests = [t.strip() for t in args.tests.split(",")] if args.tests else None

    # Quick mode overrides
    if args.quick:
        categories = QUICK_TEST_CATEGORIES
        args.attempts = 1

    # Build list of all configurations to test
    configurations: list[tuple[str, ModelConfig]] = []
    for strategy in strategies:
        for model_config in model_configs:
            configurations.append((strategy, model_config))

    print("=" * 70, flush=True)
    log("PROMPT STRATEGY & MODEL TEST HARNESS")
    print("=" * 70, flush=True)
    log(f"Strategies: {strategies}")
    log(f"Model configs: {[m.name for m in model_configs]}")
    log(f"Total configurations to test: {len(configurations)}")
    log(f"Categories: {categories or 'all'}")
    log(f"Tests: {tests or 'all'}")
    log(f"Attempts per test: {args.attempts}")
    log(f"Verbose output: {verbose}")
    print(flush=True)

    results: dict[str, BenchmarkResult] = {}
    all_test_names: set[str] = set()
    config_names: list[str] = []

    if args.compare_only:
        # Just compare existing reports
        print("Compare-only mode: analyzing existing reports...")
        strategy_reports = find_strategy_reports()

        for strategy, model_config in configurations:
            config_name = (
                f"{strategy}/{model_config.name}"
                if model_config.name != "default"
                else strategy
            )
            config_names.append(config_name)

            # Look for reports matching this config
            reports = strategy_reports.get(strategy, [])
            # Also check for model-specific reports
            if model_config.name != "default":
                model_reports = strategy_reports.get(
                    f"{strategy}_{model_config.name}", []
                )
                reports.extend(model_reports)

            if reports:
                # Use most recent report
                latest = sorted(reports, key=lambda p: p.name, reverse=True)[0]
                result = parse_report(latest, strategy, model_config)
                results[config_name] = result
                all_test_names.update(result.test_results.keys())
                print(
                    f"  {config_name}: {result.tests_passed}/{result.tests_run} passed"
                )
            else:
                print(f"  {config_name}: no reports found")

    else:
        # Run benchmarks for each configuration
        total_configs = len(configurations)
        harness_start = time.time()

        for idx, (strategy, model_config) in enumerate(configurations, 1):
            config_name = (
                f"{strategy}/{model_config.name}"
                if model_config.name != "default"
                else strategy
            )
            config_names.append(config_name)

            log(f"Progress: {idx}/{total_configs} configurations")

            result = run_benchmark_config(
                strategy=strategy,
                model_config=model_config,
                port=args.port,
                categories=categories,
                tests=tests,
                attempts=args.attempts,
                verbose=verbose,
                show_agent_output=args.agent_output,
            )
            if result:
                results[config_name] = result
                all_test_names.update(result.test_results.keys())

        # Log total harness time
        total_elapsed = time.time() - harness_start
        total_mins = total_elapsed / 60
        log(f"All benchmarks completed in {total_elapsed:.1f}s ({total_mins:.1f}m)")

    # Generate comparison report
    comparison = ComparisonReport(
        timestamp=datetime.now().isoformat(),
        configurations=config_names,
        strategies=strategies,  # For backwards compatibility
        results=results,
        test_names=sorted(all_test_names),
    )

    # Print results
    print_comparison_table(comparison)

    # Save JSON report if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(comparison.to_dict(), f, indent=2)
        print(f"\nComparison report saved to: {output_path}")

    # Also save to reports directory
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    comparison_file = (
        reports_dir / f"strategy_comparison_{datetime.now():%Y%m%dT%H%M%S}.json"
    )
    with open(comparison_file, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2)
    print(f"Comparison report saved to: {comparison_file}")

    # Run failure analysis (unless --no-analyze)
    if not args.no_analyze:
        run_failure_analysis(reports_dir)

    # Return exit code based on results
    if not results:
        sys.exit(1)

    # Success if at least one strategy had some passes
    if any(r.tests_passed > 0 for r in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
