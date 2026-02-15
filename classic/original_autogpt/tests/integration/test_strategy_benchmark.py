"""Pytest wrapper for direct_benchmark harness.

This provides CI-friendly integration of the direct_benchmark harness,
allowing it to be run as part of the pytest suite.

Usage:
    # Run tests that don't need an agent (--help, invalid args, etc.)
    poetry run pytest tests/integration/test_strategy_benchmark.py \
        -v -k "help or invalid"

    # Run full tests (requires API keys and agent to be configured)
    poetry run pytest tests/integration/test_strategy_benchmark.py -v

    # Run only specific test functions
    poetry run pytest tests/integration/test_strategy_benchmark.py::test_harness_help -v
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Mark as slow since it starts agents and runs benchmarks
pytestmark = [pytest.mark.slow, pytest.mark.integration]


def has_api_keys() -> bool:
    """Check if required API keys are configured.

    Note: When running under pytest, importing autogpt modules loads .env file,
    so this will return True if .env contains API keys.
    """
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )


# Skip condition for tests that require a running agent with API keys.
# Note: These tests also require the agent infrastructure (workspace, etc.)
# to be properly configured. They may fail even with API keys if the
# agent cannot start.
requires_agent = pytest.mark.skipif(
    not has_api_keys(),
    reason="Requires API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY)",
)


def get_direct_benchmark_dir() -> Path:
    """Get the direct_benchmark directory."""
    return Path(__file__).parent.parent.parent.parent / "direct_benchmark"


def run_harness(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run the direct_benchmark harness with given arguments.

    Args:
        *args: Arguments to pass to direct_benchmark run command
        timeout: Timeout in seconds (default: 10 minutes)

    Returns:
        CompletedProcess with stdout/stderr captured
    """
    cmd = [sys.executable, "-m", "direct_benchmark", "run", *args]
    return subprocess.run(
        cmd,
        cwd=get_direct_benchmark_dir(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@requires_agent
def test_strategy_comparison_quick():
    """Run quick strategy comparison as CI smoke test.

    This test:
    1. Starts the agent with one_shot strategy
    2. Runs general category tests
    3. Verifies at least one test produces passing results

    Note: Requires API keys to be configured in environment.
    """
    result = run_harness(
        "--fresh",  # Don't resume from previous runs
        "--strategies",
        "one_shot",
        "--categories",
        "general",
        "-N",
        "1",
        "--tests",
        "ReadFile",  # Single fast test for smoke testing
    )

    # Print output for debugging
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    assert result.returncode == 0, (
        f"Strategy benchmark failed with exit code {result.returncode}\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )


@requires_agent
def test_single_strategy():
    """Test running a single strategy with coding tests.

    This is a more focused test that only runs one_shot strategy
    to verify basic functionality without testing all strategies.
    """
    result = run_harness(
        "--fresh",  # Don't resume from previous runs
        "--strategies",
        "one_shot",
        "--categories",
        "coding",
        "--tests",
        "ReadFile,WriteFile",
    )

    # Print output for debugging
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    assert result.returncode == 0, (
        f"Single strategy test failed with exit code {result.returncode}\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )


def test_harness_help():
    """Verify the harness CLI is functional."""
    result = run_harness("--help", timeout=30)

    assert result.returncode == 0, "Harness --help should return 0"
    assert "strategies" in result.stdout.lower(), "Help should mention strategies"
    assert "categories" in result.stdout.lower(), "Help should mention categories"


def test_harness_invalid_strategy():
    """Verify the harness handles invalid strategies correctly."""
    result = run_harness("--strategies", "invalid_strategy", timeout=30)

    assert result.returncode != 0, "Invalid strategy should return non-zero"
    # Error message may be in stdout or stderr depending on the CLI framework
    combined_output = (result.stdout + result.stderr).lower()
    assert "invalid" in combined_output, "Should mention invalid strategy"
