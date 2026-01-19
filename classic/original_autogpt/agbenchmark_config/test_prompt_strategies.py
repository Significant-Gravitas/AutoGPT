#!/usr/bin/env python3
"""Test harness for comparing different prompt strategies.

This script runs the agbenchmark against the AutoGPT agent with different
prompt strategies and compares the results.

Usage:
    # Run all strategies on all tests (takes a long time)
    python test_prompt_strategies.py

    # Run specific strategies
    python test_prompt_strategies.py --strategies one_shot,plan_execute

    # Run specific test categories
    python test_prompt_strategies.py --categories code,retrieval

    # Run specific tests
    python test_prompt_strategies.py --tests TestWriteFile,TestReadFile

    # Quick smoke test (1 attempt, basic tests only)
    python test_prompt_strategies.py --quick

    # Compare existing reports without running new tests
    python test_prompt_strategies.py --compare-only

    # Run with multiple attempts per test
    python test_prompt_strategies.py --attempts 3
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
class StrategyResult:
    """Results for a single strategy run."""

    strategy: str
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


@dataclass
class ComparisonReport:
    """Comparison report across all strategies."""

    timestamp: str
    strategies: list[str]
    results: dict[str, StrategyResult]
    test_names: list[str]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "strategies": self.strategies,
            "results": {
                name: {
                    "strategy": r.strategy,
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


def start_agent(strategy: str, port: int = 8000) -> subprocess.Popen:
    """Start the AutoGPT agent with a specific strategy."""
    env = os.environ.copy()
    env["PROMPT_STRATEGY"] = strategy
    env["AP_SERVER_PORT"] = str(port)

    print(f"  Starting agent with strategy '{strategy}' on port {port}...")

    # Start the agent server (port is set via AP_SERVER_PORT env var)
    proc = subprocess.Popen(
        ["poetry", "run", "serve"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent.parent,
    )

    # Wait for agent to be ready
    start_time = time.time()
    while time.time() - start_time < AGENT_STARTUP_TIMEOUT:
        try:
            import urllib.request

            urllib.request.urlopen(f"http://localhost:{port}/ap/v1/agent/tasks")
            print(f"  Agent ready on port {port}")
            return proc
        except Exception:
            time.sleep(0.5)

    proc.kill()
    raise TimeoutError(f"Agent failed to start within {AGENT_STARTUP_TIMEOUT}s")


def stop_agent(proc: subprocess.Popen) -> None:
    """Stop the agent process."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_benchmark(
    strategy: str,
    port: int = 8000,
    categories: Optional[list[str]] = None,
    tests: Optional[list[str]] = None,
    attempts: int = 1,
    timeout: int = BENCHMARK_TIMEOUT,
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

    print(f"  Running benchmark: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=Path(__file__).parent.parent,
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Non-zero exit is normal - agbenchmark returns non-zero when tests fail
            print(f"  Benchmark completed with code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            # Continue to look for report - tests may have run

    except subprocess.TimeoutExpired:
        print(f"  Benchmark timed out after {timeout}s")
        return None

    # Find the most recent report directory
    reports_dir = Path(__file__).parent / "reports"
    if not reports_dir.exists():
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
        return report_dirs[0]
    return None


def parse_report(report_dir: Path, strategy: str) -> StrategyResult:
    """Parse a benchmark report and extract metrics."""
    result = StrategyResult(strategy=strategy, report_dir=report_dir)

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
            success = metrics.get("success", False)

            if success:
                result.tests_passed += 1
            else:
                result.tests_failed += 1

            # Get detailed results if available
            test_results = test_data.get("results", [])
            if test_results:
                first_result = test_results[0]
                result.total_time += float(
                    first_result.get("run_time", "0").rstrip("s")
                )
                result.total_cost += first_result.get("cost", 0) or 0
                n_steps = first_result.get("n_steps", 0)
            else:
                n_steps = 0

            result.test_results[full_name] = {
                "success": success,
                "n_steps": n_steps,
                "difficulty": test_data.get("difficulty", "unknown"),
            }

    process_tests(test_tree)

    if result.tests_run > 0:
        total_steps = sum(t.get("n_steps", 0) for t in result.test_results.values())
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
    print("PROMPT STRATEGY COMPARISON REPORT")
    print("=" * 80)
    print(f"Timestamp: {report.timestamp}")
    print()

    # Summary table
    print("SUMMARY")
    print("-" * 80)
    headers = [
        "Strategy",
        "Tests",
        "Passed",
        "Failed",
        "Success %",
        "Avg Steps",
        "Cost",
    ]
    rows = []

    for strategy in report.strategies:
        r = report.results.get(strategy)
        if r:
            rows.append(
                [
                    strategy,
                    r.tests_run,
                    r.tests_passed,
                    r.tests_failed,
                    f"{r.success_rate:.1f}%",
                    f"{r.avg_steps:.1f}",
                    f"${r.total_cost:.4f}",
                ]
            )
        else:
            rows.append([strategy, "-", "-", "-", "-", "-", "-"])

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

        test_headers = ["Test"] + report.strategies
        test_rows = []

        for test_name in sorted(report.test_names):
            row = [test_name[:40]]  # Truncate long names
            for strategy in report.strategies:
                r = report.results.get(strategy)
                if r and test_name in r.test_results:
                    tr = r.test_results[test_name]
                    status = "✅" if tr["success"] else "❌"
                    row.append(f"{status} ({tr['n_steps']} steps)")
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


def run_strategy_benchmark(
    strategy: str,
    port: int,
    categories: Optional[list[str]],
    tests: Optional[list[str]],
    attempts: int,
) -> Optional[StrategyResult]:
    """Run benchmark for a single strategy."""
    print(f"\n{'='*60}")
    print(f"Testing strategy: {strategy}")
    print("=" * 60)

    agent_proc = None
    try:
        # Start agent
        agent_proc = start_agent(strategy, port)

        # Run benchmark
        report_dir = run_benchmark(
            strategy=strategy,
            port=port,
            categories=categories,
            tests=tests,
            attempts=attempts,
        )

        if report_dir:
            # Rename report directory to include strategy name
            new_name = f"{report_dir.name}_{strategy}"
            new_path = report_dir.parent / new_name
            if not new_path.exists():
                report_dir.rename(new_path)
                report_dir = new_path

            # Parse results
            result = parse_report(report_dir, strategy)
            print(f"  Results: {result.tests_passed}/{result.tests_run} passed")
            return result
        else:
            print("  No report generated")
            return None

    except Exception as e:
        print(f"  Error: {e}")
        return None

    finally:
        if agent_proc:
            stop_agent(agent_proc)


def main():
    parser = argparse.ArgumentParser(
        description="Test harness for comparing prompt strategies"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        help=f"Comma-separated list of strategies to test (default: all). "
        f"Available: {', '.join(STRATEGIES)}",
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
        "--compare-only",
        action="store_true",
        help="Only compare existing reports, don't run new tests",
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

    args = parser.parse_args()

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

    # Parse categories/tests
    categories = (
        [c.strip() for c in args.categories.split(",")] if args.categories else None
    )
    tests = [t.strip() for t in args.tests.split(",")] if args.tests else None

    # Quick mode overrides
    if args.quick:
        categories = QUICK_TEST_CATEGORIES
        args.attempts = 1

    print("=" * 60)
    print("PROMPT STRATEGY TEST HARNESS")
    print("=" * 60)
    print(f"Strategies to test: {strategies}")
    print(f"Categories: {categories or 'all'}")
    print(f"Tests: {tests or 'all'}")
    print(f"Attempts per test: {args.attempts}")
    print()

    results: dict[str, StrategyResult] = {}
    all_test_names: set[str] = set()

    if args.compare_only:
        # Just compare existing reports
        print("Compare-only mode: analyzing existing reports...")
        strategy_reports = find_strategy_reports()

        for strategy in strategies:
            reports = strategy_reports.get(strategy, [])
            if reports:
                # Use most recent report
                latest = sorted(reports, key=lambda p: p.name, reverse=True)[0]
                result = parse_report(latest, strategy)
                results[strategy] = result
                all_test_names.update(result.test_results.keys())
                print(f"  {strategy}: {result.tests_passed}/{result.tests_run} passed")
            else:
                print(f"  {strategy}: no reports found")

    else:
        # Run benchmarks for each strategy
        for strategy in strategies:
            result = run_strategy_benchmark(
                strategy=strategy,
                port=args.port,
                categories=categories,
                tests=tests,
                attempts=args.attempts,
            )
            if result:
                results[strategy] = result
                all_test_names.update(result.test_results.keys())

    # Generate comparison report
    comparison = ComparisonReport(
        timestamp=datetime.now().isoformat(),
        strategies=strategies,
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
