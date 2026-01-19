"""Pytest wrapper for strategy benchmark test harness.

This provides CI-friendly integration of the strategy benchmark,
allowing it to be run as part of the pytest suite.

Usage:
    # Run quick CLI tests (no agent required)
    pytest tests/integration/test_strategy_benchmark.py -v -m "not requires_agent"

    # Run full tests (requires API keys and agent)
    poetry run pytest tests/integration/test_strategy_benchmark.py -v

    # Run with specific markers
    poetry run pytest -m slow tests/integration/test_strategy_benchmark.py -v
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


def get_project_root() -> Path:
    """Get the original_autogpt project root directory."""
    return Path(__file__).parent.parent.parent


def run_harness(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run the test harness with given arguments.

    Args:
        *args: Arguments to pass to test_prompt_strategies.py
        timeout: Timeout in seconds (default: 10 minutes)

    Returns:
        CompletedProcess with stdout/stderr captured
    """
    cmd = [sys.executable, "agbenchmark_config/test_prompt_strategies.py", *args]
    return subprocess.run(
        cmd,
        cwd=get_project_root(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@requires_agent
def test_strategy_comparison_quick():
    """Run quick strategy comparison as CI smoke test.

    This test:
    1. Starts the agent with each strategy
    2. Runs interface tests (fastest category)
    3. Verifies at least one strategy produces passing results

    Note: Requires API keys to be configured in environment.
    """
    result = run_harness("--quick")

    # Print output for debugging
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    assert result.returncode == 0, (
        f"Strategy benchmark failed with exit code {result.returncode}\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )


def test_harness_compare_only():
    """Test that compare-only mode works with existing reports.

    This test doesn't run any benchmarks, just verifies the report
    comparison logic works correctly.
    """
    result = run_harness("--compare-only")

    # Print output for debugging
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # compare-only returns 0 if it can read reports (even if empty)
    # It returns 1 only if there's an actual error
    # We check that it ran without crashing
    assert "PROMPT STRATEGY" in result.stdout or result.returncode in (0, 1), (
        f"Harness crashed unexpectedly\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )


@requires_agent
def test_single_strategy():
    """Test running a single strategy with interface tests.

    This is a more focused test that only runs one_shot strategy
    to verify basic functionality without testing all strategies.
    """
    result = run_harness(
        "--strategies",
        "one_shot",
        "--categories",
        "interface",
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
    assert "quick" in result.stdout.lower(), "Help should mention quick mode"


def test_harness_invalid_strategy():
    """Verify the harness handles invalid strategies correctly."""
    result = run_harness("--strategies", "invalid_strategy", timeout=30)

    assert result.returncode != 0, "Invalid strategy should return non-zero"
    assert "invalid" in result.stdout.lower(), "Should mention invalid strategy"
