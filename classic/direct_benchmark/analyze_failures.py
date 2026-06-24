#!/usr/bin/env python3
"""
Strategy Failure Analysis Tool

Analyzes why prompt strategies fail on benchmark tests, identifies patterns,
and provides actionable insights for improvement.

Usage:
    # Full analysis with LLM summaries (default)
    poetry run python agbenchmark_config/analyze_failures.py

    # Disable LLM analysis (just print raw pattern data)
    poetry run python agbenchmark_config/analyze_failures.py --no-analysis

    # Focus on specific strategy
    poetry run python agbenchmark_config/analyze_failures.py --strategy rewoo

    # Compare one test across strategies (interactive)
    poetry run python agbenchmark_config/analyze_failures.py --test Battleship

    # Interactive drill-down mode
    poetry run python agbenchmark_config/analyze_failures.py --interactive

    # Export to markdown
    poetry run python agbenchmark_config/analyze_failures.py --markdown
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Type hints for optional rich imports
Console: Any = None
Markdown: Any = None
Panel: Any = None
Progress: Any = None
SpinnerColumn: Any = None
TextColumn: Any = None
Confirm: Any = None
Prompt: Any = None
Table: Any = None
Text: Any = None
Tree: Any = None

try:
    from rich.console import Console
    from rich.markdown import Markdown  # noqa: F401
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt  # noqa: F401
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class FailurePattern(Enum):
    """Categories of failure patterns."""

    OVER_PLANNING = "over_planning"  # Too many planning steps, not enough execution
    TOOL_LOOP = "tool_loop"  # Repeating same tool without progress
    MISSING_CRITICAL = "missing_critical"  # Didn't complete key action
    TIMEOUT = "timeout"  # Hit step limit before completion
    ERROR_UNRECOVERED = "error_unrecovered"  # Hit error and couldn't recover
    WRONG_APPROACH = "wrong_approach"  # Fundamentally wrong solution
    UNKNOWN = "unknown"


@dataclass
class StepInfo:
    """Information about a single execution step."""

    step_num: int
    tool_name: str
    tool_args: dict
    tool_result: Optional[dict]
    thoughts: dict
    cumulative_cost: float
    output: str


@dataclass
class TestResult:
    """Analysis of a single test execution."""

    test_name: str
    strategy: str
    task: str
    success: bool
    fail_reason: Optional[str]
    reached_cutoff: bool
    n_steps: int
    steps: list[StepInfo]
    total_cost: float
    run_time: str
    tool_distribution: Counter = field(default_factory=Counter)
    patterns_detected: list[FailurePattern] = field(default_factory=list)


@dataclass
class StrategyAnalysis:
    """Analysis results for a strategy."""

    strategy_name: str
    total_tests: int
    passed: int
    failed: int
    success_rate: float
    total_cost: float
    avg_steps: float
    failed_tests: list[TestResult]
    pattern_distribution: Counter = field(default_factory=Counter)


class FailureAnalyzer:
    """Main analysis engine."""

    def __init__(self, reports_dir: Path, use_llm: bool = True):
        self.reports_dir = reports_dir
        self.use_llm = use_llm
        self._console_instance = Console() if RICH_AVAILABLE else None
        self.strategies: dict[str, StrategyAnalysis] = {}
        self.test_comparison: dict[str, dict[str, TestResult]] = defaultdict(dict)
        self._llm_provider = None

    @property
    def console(self) -> Any:
        """Get console instance (only call when RICH_AVAILABLE is True)."""
        assert self._console_instance is not None
        return self._console_instance

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Print with Rich if available, otherwise standard print."""
        if self._console_instance:
            self._console_instance.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def find_reports(self) -> list[tuple[str, Path]]:
        """Find all strategy-specific reports."""
        reports = []
        for report_dir in self.reports_dir.iterdir():
            if not report_dir.is_dir():
                continue
            report_file = report_dir / "report.json"
            if not report_file.exists():
                continue

            # Extract strategy from directory name
            name = report_dir.name
            strategy = None
            for s in [
                "one_shot",
                "rewoo",
                "plan_execute",
                "reflexion",
                "tree_of_thoughts",
            ]:
                if s in name:
                    strategy = s
                    break

            if strategy:
                reports.append((strategy, report_file))

        return sorted(reports, key=lambda x: x[1].stat().st_mtime, reverse=True)

    def parse_report(self, strategy: str, report_path: Path) -> StrategyAnalysis:
        """Parse a benchmark report file."""
        with open(report_path) as f:
            data = json.load(f)

        tests_data = data.get("tests", {})
        failed_tests = []
        total_cost = 0.0
        total_steps = 0
        passed = 0
        failed = 0

        for test_name, test_data in tests_data.items():
            results = test_data.get("results", [])
            if not results:
                continue

            result = results[0]
            success = result.get("success", False)
            n_steps = result.get("n_steps", 0)
            cost = result.get("cost", 0)

            total_steps += n_steps
            total_cost += cost or 0

            if success:
                passed += 1
            else:
                failed += 1
                test_result = self._parse_test_result(
                    test_name, strategy, test_data, result
                )
                failed_tests.append(test_result)
                self.test_comparison[test_name][strategy] = test_result

        total_tests = passed + failed
        return StrategyAnalysis(
            strategy_name=strategy,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            success_rate=(passed / total_tests * 100) if total_tests > 0 else 0,
            total_cost=total_cost,
            avg_steps=total_steps / total_tests if total_tests > 0 else 0,
            failed_tests=failed_tests,
        )

    def _parse_test_result(
        self, test_name: str, strategy: str, test_data: dict, result: dict
    ) -> TestResult:
        """Parse a single test result."""
        steps_data = result.get("steps", [])
        steps = []
        tool_distribution = Counter()

        for i, step in enumerate(steps_data):
            ao = step.get("additional_output") or {}
            use_tool = ao.get("use_tool") or {}
            last_action = ao.get("last_action") or {}
            thoughts = ao.get("thoughts") or {}

            tool_name = use_tool.get("name", "none")
            tool_distribution[tool_name] += 1

            step_info = StepInfo(
                step_num=i + 1,
                tool_name=tool_name,
                tool_args=use_tool.get("arguments", {}),
                tool_result=last_action.get("result") if last_action else None,
                thoughts=thoughts,
                cumulative_cost=ao.get("task_cumulative_cost", 0),
                output=step.get("output", ""),
            )
            steps.append(step_info)

        test_result = TestResult(
            test_name=test_name,
            strategy=strategy,
            task=test_data.get("task", ""),
            success=False,
            fail_reason=result.get("fail_reason"),
            reached_cutoff=result.get("reached_cutoff", False),
            n_steps=result.get("n_steps", 0),
            steps=steps,
            total_cost=result.get("cost", 0),
            run_time=result.get("run_time", ""),
            tool_distribution=tool_distribution,
        )

        # Detect patterns
        test_result.patterns_detected = self._detect_patterns(test_result)
        return test_result

    def _detect_patterns(self, test: TestResult) -> list[FailurePattern]:
        """Detect failure patterns in a test result."""
        patterns = []

        # Pattern 1: Over-planning
        planning_tools = {"todo_write", "todo_read", "think", "plan"}
        execution_tools = {
            "write_file",
            "execute_python",
            "execute_shell",
            "read_file",
        }

        planning_count = sum(test.tool_distribution.get(t, 0) for t in planning_tools)
        _execution_count = sum(  # noqa: F841
            test.tool_distribution.get(t, 0) for t in execution_tools
        )

        if test.n_steps > 0:
            planning_ratio = planning_count / test.n_steps
            if planning_ratio > 0.5 and test.n_steps > 1:
                patterns.append(FailurePattern.OVER_PLANNING)

        # Pattern 2: Tool loops (same tool used 3+ times consecutively)
        if len(test.steps) >= 3:
            for i in range(len(test.steps) - 2):
                if (
                    test.steps[i].tool_name
                    == test.steps[i + 1].tool_name
                    == test.steps[i + 2].tool_name
                ):
                    patterns.append(FailurePattern.TOOL_LOOP)
                    break

        # Pattern 3: Missing critical action
        # If task mentions "write" or "create" but no write_file was used
        task_lower = test.task.lower()
        if any(word in task_lower for word in ["write", "create", "generate", "build"]):
            if test.tool_distribution.get("write_file", 0) == 0:
                patterns.append(FailurePattern.MISSING_CRITICAL)

        # Pattern 4: Timeout
        if test.reached_cutoff:
            patterns.append(FailurePattern.TIMEOUT)

        # Pattern 5: Error unrecovered
        error_count = 0
        for step in test.steps:
            if step.tool_result and step.tool_result.get("status") == "error":
                error_count += 1
        if error_count > 0 and error_count == len(test.steps) - 1:
            patterns.append(FailurePattern.ERROR_UNRECOVERED)

        if not patterns:
            patterns.append(FailurePattern.UNKNOWN)

        return patterns

    def analyze_all(self) -> None:
        """Analyze all available reports."""
        reports = self.find_reports()

        # Keep only most recent report per strategy
        latest_reports = {}
        for strategy, path in reports:
            if strategy not in latest_reports:
                latest_reports[strategy] = path

        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "Analyzing reports...", total=len(latest_reports)
                )
                for strategy, path in latest_reports.items():
                    progress.update(task, description=f"Analyzing {strategy}...")
                    self.strategies[strategy] = self.parse_report(strategy, path)
                    progress.advance(task)
        else:
            for strategy, path in latest_reports.items():
                print(f"Analyzing {strategy}...")
                self.strategies[strategy] = self.parse_report(strategy, path)

    def _get_llm_provider(self) -> Any:
        """Lazy-load the LLM provider."""
        if self._llm_provider is None:
            try:
                # Add parent paths to find forge
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / "forge"))
                from forge.llm.providers import MultiProvider

                self._llm_provider = MultiProvider()
            except ImportError as e:
                self._print(
                    f"[yellow]Warning: Could not load LLM provider: {e}[/yellow]"
                    if RICH_AVAILABLE
                    else f"Warning: Could not load LLM provider: {e}"
                )
                self._llm_provider = False
        return self._llm_provider if self._llm_provider else None

    async def _get_llm_analysis(self, test: TestResult) -> Optional[str]:
        """Get LLM-powered analysis of a failure.

        Note: This is a placeholder for future LLM-powered analysis.
        Currently disabled to avoid dependency issues.
        """
        # LLM analysis disabled for now - patterns provide sufficient insights
        return None

    def print_summary(self) -> None:
        """Print overall summary."""
        if RICH_AVAILABLE:
            table = Table(title="Strategy Comparison Summary")
            table.add_column("Strategy", style="cyan")
            table.add_column("Tests", justify="right")
            table.add_column("Passed", justify="right", style="green")
            table.add_column("Failed", justify="right", style="red")
            table.add_column("Success %", justify="right")
            table.add_column("Avg Steps", justify="right")
            table.add_column("Cost", justify="right")

            for name, analysis in sorted(
                self.strategies.items(), key=lambda x: x[1].success_rate, reverse=True
            ):
                table.add_row(
                    name,
                    str(analysis.total_tests),
                    str(analysis.passed),
                    str(analysis.failed),
                    f"{analysis.success_rate:.1f}%",
                    f"{analysis.avg_steps:.1f}",
                    f"${analysis.total_cost:.4f}",
                )

            self.console.print(table)
        else:
            print("\n=== Strategy Comparison Summary ===")
            hdr = (
                f"{'Strategy':<20} {'Tests':>6} {'Passed':>7} "
                f"{'Failed':>7} {'Success%':>10} {'AvgSteps':>9} {'Cost':>10}"
            )
            print(hdr)
            print("-" * 80)
            for name, analysis in sorted(
                self.strategies.items(), key=lambda x: x[1].success_rate, reverse=True
            ):
                row = (
                    f"{name:<20} {analysis.total_tests:>6} "
                    f"{analysis.passed:>7} {analysis.failed:>7} "
                    f"{analysis.success_rate:>9.1f}% {analysis.avg_steps:>9.1f} "
                    f"${analysis.total_cost:>9.4f}"
                )
                print(row)

    def print_pattern_analysis(self) -> None:
        """Print failure pattern analysis."""
        all_patterns = Counter()
        for analysis in self.strategies.values():
            for test in analysis.failed_tests:
                for pattern in test.patterns_detected:
                    all_patterns[pattern] += 1

        self._print("\n")
        if RICH_AVAILABLE:
            table = Table(title="Failure Pattern Distribution")
            table.add_column("Pattern", style="yellow")
            table.add_column("Count", justify="right")
            table.add_column("Description")

            pattern_descriptions = {
                FailurePattern.OVER_PLANNING: "Too much planning, not enough action",
                FailurePattern.TOOL_LOOP: "Repeats same tool 3+ times consecutively",
                FailurePattern.MISSING_CRITICAL: "Never performed key action",
                FailurePattern.TIMEOUT: "Hit step limit before completing task",
                FailurePattern.ERROR_UNRECOVERED: "Hit errors and couldn't recover",
                FailurePattern.WRONG_APPROACH: "Took fundamentally wrong approach",
                FailurePattern.UNKNOWN: "Pattern not categorized",
            }

            for pattern, count in all_patterns.most_common():
                table.add_row(
                    pattern.value, str(count), pattern_descriptions.get(pattern, "")
                )

            self.console.print(table)
        else:
            print("\n=== Failure Pattern Distribution ===")
            for pattern, count in all_patterns.most_common():
                print(f"  {pattern.value}: {count}")

    def print_failed_tests(self, strategy: Optional[str] = None) -> None:
        """Print detailed failure analysis."""
        strategies_to_show = (
            [self.strategies[strategy]] if strategy else self.strategies.values()
        )

        for analysis in strategies_to_show:
            self._print("\n")
            if RICH_AVAILABLE:
                msg = (
                    f"[bold]{analysis.strategy_name}[/bold] - "
                    f"{analysis.failed} failures out of {analysis.total_tests} tests"
                )
                self.console.print(Panel(msg, title="Strategy Analysis"))
            else:
                print(f"\n=== {analysis.strategy_name} ===")
                print(f"Failures: {analysis.failed}/{analysis.total_tests}")

            for test in analysis.failed_tests:
                self._print_test_failure(test)

    def _print_test_failure(self, test: TestResult) -> None:
        """Print a single test failure."""
        if RICH_AVAILABLE:
            tree = Tree(f"[red]{test.test_name}[/red]")
            tree.add(f"[dim]Task:[/dim] {test.task[:80]}...")
            tree.add(f"[dim]Steps:[/dim] {test.n_steps}")
            tree.add(f"[dim]Cost:[/dim] ${test.total_cost:.4f}")
            patterns = ", ".join(p.value for p in test.patterns_detected)
            tree.add(f"[dim]Patterns:[/dim] {patterns}")

            tools = tree.add("[dim]Tool sequence:[/dim]")
            tool_seq = [s.tool_name for s in test.steps[:10]]
            tools.add(" -> ".join(tool_seq) + ("..." if len(test.steps) > 10 else ""))

            if test.fail_reason:
                reason = tree.add("[dim]Fail reason:[/dim]")
                reason.add(Text(test.fail_reason[:200], style="red"))

            self.console.print(tree)
        else:
            print(f"\n  {test.test_name}")
            print(f"    Task: {test.task[:80]}...")
            print(f"    Steps: {test.n_steps}, Cost: ${test.total_cost:.4f}")
            print(f"    Patterns: {', '.join(p.value for p in test.patterns_detected)}")
            tool_seq = [s.tool_name for s in test.steps[:10]]
            print(f"    Tools: {' -> '.join(tool_seq)}")
            if test.fail_reason:
                print(f"    Fail reason: {test.fail_reason[:200]}")

    def compare_test(self, test_name: str) -> None:
        """Compare a single test across all strategies."""
        if test_name not in self.test_comparison:
            self._print(
                f"[red]Test '{test_name}' not found in failed tests[/red]"
                if RICH_AVAILABLE
                else f"Test '{test_name}' not found in failed tests"
            )
            return

        results = self.test_comparison[test_name]
        self._print("\n")
        if RICH_AVAILABLE:
            self.console.print(Panel(f"[bold]Comparing: {test_name}[/bold]"))
        else:
            print(f"\n=== Comparing: {test_name} ===")

        for strategy, test in sorted(results.items()):
            self._print("\n")
            if RICH_AVAILABLE:
                self.console.print(f"[cyan]--- {strategy} ---[/cyan]")
            else:
                print(f"\n--- {strategy} ---")
            self._print_test_failure(test)

    def interactive_mode(self) -> None:
        """Run interactive exploration mode."""
        if not RICH_AVAILABLE:
            print("Interactive mode requires the 'rich' library.")
            print("Install with: pip install rich")
            return

        while True:
            self.console.print("\n[bold]Interactive Failure Analysis[/bold]")
            self.console.print("Commands:")
            self.console.print("  [cyan]summary[/cyan] - Show overall summary")
            self.console.print("  [cyan]patterns[/cyan] - Show pattern analysis")
            self.console.print(
                "  [cyan]strategy <name>[/cyan] - Show failures for a strategy"
            )
            self.console.print(
                "  [cyan]test <name>[/cyan] - Compare test across strategies"
            )
            self.console.print(
                "  [cyan]step <strategy> <test> <n>[/cyan] - Show step details"
            )
            self.console.print("  [cyan]list tests[/cyan] - List all failed tests")
            self.console.print("  [cyan]list strategies[/cyan] - List strategies")
            self.console.print("  [cyan]quit[/cyan] - Exit")

            cmd = Prompt.ask("\n[bold]>>[/bold]").strip().lower()

            if cmd == "quit" or cmd == "q":
                break
            elif cmd == "summary":
                self.print_summary()
            elif cmd == "patterns":
                self.print_pattern_analysis()
            elif cmd.startswith("strategy "):
                strategy = cmd.split(" ", 1)[1]
                if strategy in self.strategies:
                    self.print_failed_tests(strategy)
                else:
                    self.console.print(f"[red]Unknown strategy: {strategy}[/red]")
            elif cmd.startswith("test "):
                test_name = cmd.split(" ", 1)[1]
                self.compare_test(test_name)
            elif cmd.startswith("step "):
                parts = cmd.split()
                if len(parts) >= 4:
                    strategy = parts[1]
                    test_name = parts[2]
                    step_num = int(parts[3])
                    self._show_step_detail(strategy, test_name, step_num)
                else:
                    self.console.print(
                        "[red]Usage: step <strategy> <test> <step_num>[/red]"
                    )
            elif cmd == "list tests":
                self._list_tests()
            elif cmd == "list strategies":
                self.console.print(", ".join(self.strategies.keys()))
            else:
                self.console.print(f"[red]Unknown command: {cmd}[/red]")

    def _list_tests(self) -> None:
        """List all failed tests."""
        all_tests = set()
        for analysis in self.strategies.values():
            for test in analysis.failed_tests:
                all_tests.add(test.test_name)

        if RICH_AVAILABLE:
            table = Table(title="Failed Tests Across Strategies")
            table.add_column("Test", style="cyan")
            for strategy in self.strategies.keys():
                table.add_column(strategy, justify="center")

            for test_name in sorted(all_tests):
                row = [test_name]
                for strategy in self.strategies.keys():
                    if (
                        test_name in self.test_comparison
                        and strategy in self.test_comparison[test_name]
                    ):
                        row.append("[red]FAIL[/red]")
                    else:
                        row.append("[green]PASS[/green]")
                table.add_row(*row)

            self.console.print(table)
        else:
            print("\n=== Failed Tests ===")
            for test_name in sorted(all_tests):
                print(f"  {test_name}")

    def _show_step_detail(self, strategy: str, test_name: str, step_num: int) -> None:
        """Show detailed information about a specific step."""
        if strategy not in self.strategies:
            self._print(
                f"[red]Unknown strategy: {strategy}[/red]"
                if RICH_AVAILABLE
                else f"Unknown strategy: {strategy}"
            )
            return

        test = None
        for t in self.strategies[strategy].failed_tests:
            if t.test_name == test_name:
                test = t
                break

        if not test:
            self._print(
                f"[red]Test '{test_name}' not found in {strategy}[/red]"
                if RICH_AVAILABLE
                else f"Test '{test_name}' not found in {strategy}"
            )
            return

        if step_num < 1 or step_num > len(test.steps):
            self._print(
                f"[red]Step {step_num} out of range (1-{len(test.steps)})[/red]"
                if RICH_AVAILABLE
                else f"Step {step_num} out of range (1-{len(test.steps)})"
            )
            return

        step = test.steps[step_num - 1]

        if RICH_AVAILABLE:
            self.console.print(Panel(f"[bold]Step {step_num} Details[/bold]"))
            self.console.print(f"[cyan]Tool:[/cyan] {step.tool_name}")
            self.console.print(
                f"[cyan]Arguments:[/cyan] {json.dumps(step.tool_args, indent=2)}"
            )

            if step.thoughts:
                self.console.print("\n[cyan]Thoughts:[/cyan]")
                for key, value in step.thoughts.items():
                    self.console.print(f"  [dim]{key}:[/dim] {value}")

            if step.tool_result:
                result_str = json.dumps(step.tool_result, indent=2)[:500]
                self.console.print(f"\n[cyan]Result:[/cyan] {result_str}")

            self.console.print(
                f"\n[cyan]Cumulative Cost:[/cyan] ${step.cumulative_cost:.4f}"
            )
        else:
            print(f"\n=== Step {step_num} Details ===")
            print(f"Tool: {step.tool_name}")
            print(f"Arguments: {json.dumps(step.tool_args, indent=2)}")
            if step.thoughts:
                print("\nThoughts:")
                for key, value in step.thoughts.items():
                    print(f"  {key}: {value}")
            if step.tool_result:
                print(f"\nResult: {json.dumps(step.tool_result, indent=2)[:500]}")
            print(f"\nCumulative Cost: ${step.cumulative_cost:.4f}")

    def export_markdown(self, output_path: Optional[Path] = None) -> str:
        """Export analysis to markdown format."""
        lines = []
        lines.append("# Benchmark Failure Analysis Report")
        lines.append(f"\nGenerated: {datetime.now().isoformat()}\n")

        # Summary table
        lines.append("## Strategy Comparison\n")
        lines.append(
            "| Strategy | Tests | Passed | Failed | Success % | Avg Steps | Cost |"
        )
        lines.append(
            "|----------|-------|--------|--------|-----------|-----------|------|"
        )
        for name, analysis in sorted(
            self.strategies.items(), key=lambda x: x[1].success_rate, reverse=True
        ):
            row = (
                f"| {name} | {analysis.total_tests} | {analysis.passed} "
                f"| {analysis.failed} | {analysis.success_rate:.1f}% "
                f"| {analysis.avg_steps:.1f} | ${analysis.total_cost:.4f} |"
            )
            lines.append(row)

        # Pattern analysis
        lines.append("\n## Failure Patterns\n")
        all_patterns = Counter()
        for analysis in self.strategies.values():
            for test in analysis.failed_tests:
                for pattern in test.patterns_detected:
                    all_patterns[pattern] += 1

        for pattern, count in all_patterns.most_common():
            lines.append(f"- **{pattern.value}**: {count} occurrences")

        # Failed tests by strategy
        lines.append("\n## Failed Tests by Strategy\n")
        for name, analysis in self.strategies.items():
            if not analysis.failed_tests:
                continue
            lines.append(f"\n### {name}\n")
            for test in analysis.failed_tests:
                lines.append(f"#### {test.test_name}\n")
                lines.append(f"- **Task**: {test.task[:100]}...")
                lines.append(f"- **Steps**: {test.n_steps}")
                patterns = ", ".join(p.value for p in test.patterns_detected)
                lines.append(f"- **Patterns**: {patterns}")
                tools = " -> ".join(s.tool_name for s in test.steps[:8])
                lines.append(f"- **Tool sequence**: {tools}")
                if test.fail_reason:
                    lines.append(f"- **Fail reason**: {test.fail_reason[:150]}...")
                lines.append("")

        content = "\n".join(lines)

        if output_path:
            output_path.write_text(content)
            self._print(
                f"Markdown report saved to: {output_path}"
                if not RICH_AVAILABLE
                else f"[green]Markdown report saved to: {output_path}[/green]"
            )

        return content


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark failures across prompt strategies"
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Disable LLM-powered analysis",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Focus on a specific strategy",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Compare a specific test across strategies",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        nargs="?",
        const="failure_analysis.md",
        help="Export to markdown (optionally specify output file)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Path to reports directory",
    )

    args = parser.parse_args()

    # Find reports directory
    if args.reports_dir:
        reports_dir = Path(args.reports_dir)
    else:
        # Try to find it relative to this script
        script_dir = Path(__file__).parent
        reports_dir = script_dir / "reports"
        if not reports_dir.exists():
            reports_dir = Path.cwd() / "agbenchmark_config" / "reports"

    if not reports_dir.exists():
        print(f"Reports directory not found: {reports_dir}")
        sys.exit(1)

    analyzer = FailureAnalyzer(reports_dir, use_llm=not args.no_analysis)
    analyzer.analyze_all()

    if not analyzer.strategies:
        print("No strategy reports found.")
        sys.exit(1)

    if args.interactive:
        analyzer.interactive_mode()
    elif args.test:
        analyzer.compare_test(args.test)
    elif args.strategy:
        analyzer.print_failed_tests(args.strategy)
    else:
        analyzer.print_summary()
        analyzer.print_pattern_analysis()
        analyzer.print_failed_tests()

    if args.markdown:
        output_path = Path(args.markdown)
        analyzer.export_markdown(output_path)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
