"""Rich UI components for the benchmark harness."""

from datetime import datetime
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .models import ChallengeResult, ExecutionProgress

console = Console()


class BenchmarkUI:
    """Rich UI for benchmark progress and results."""

    def __init__(self, max_parallel: int = 4, verbose: bool = False):
        self.max_parallel = max_parallel
        self.verbose = verbose
        self.start_time: Optional[datetime] = None

        # Track state
        self.active_runs: dict[str, str] = {}  # config_name -> challenge_name
        self.completed: list[ChallengeResult] = []
        self.results_by_config: dict[str, list[ChallengeResult]] = {}

        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self.main_task: Optional[TaskID] = None
        self.total_challenges = 0

    def start(self, total_challenges: int, configs: list[str]) -> None:
        """Start the UI with the given number of challenges."""
        self.start_time = datetime.now()
        self.total_challenges = total_challenges
        self.results_by_config = {config: [] for config in configs}
        self.main_task = self.progress.add_task(
            "[cyan]Running benchmarks...", total=total_challenges
        )

    def update(self, progress: ExecutionProgress) -> None:
        """Update UI with execution progress."""
        if progress.status == "starting":
            self.active_runs[progress.config_name] = progress.challenge_name
        elif progress.status in ("completed", "failed"):
            if progress.config_name in self.active_runs:
                del self.active_runs[progress.config_name]
            if progress.result:
                self.completed.append(progress.result)
                self.results_by_config[progress.config_name].append(progress.result)
                if self.main_task is not None:
                    self.progress.advance(self.main_task)

                if self.verbose:
                    self._print_challenge_result(progress.result)

    def _print_challenge_result(self, result: ChallengeResult) -> None:
        """Print detailed result for a single challenge."""
        status_icon = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
        console.print(
            f"[dim][{result.config_name}][/dim] {result.challenge_name}: {status_icon} "
            f"({result.n_steps} steps, ${result.cost:.4f})"
        )

    def render_active_runs(self) -> Panel:
        """Render panel showing active runs."""
        if not self.active_runs:
            content = Text("No active runs", style="dim")
        else:
            lines = []
            for config_name, challenge_name in self.active_runs.items():
                lines.append(
                    Text.assemble(
                        ("  ", ""),
                        ("\u25cf ", "yellow"),  # Bullet point
                        (f"{config_name}", "cyan"),
                        (" \u2192 ", "dim"),  # Arrow
                        (challenge_name, "white"),
                    )
                )
            content = Group(*lines)

        return Panel(
            content,
            title=f"[bold]Active Runs ({len(self.active_runs)}/{self.max_parallel})[/bold]",
            border_style="blue",
        )

    def render_summary_table(self) -> Table:
        """Render summary table of results by configuration."""
        table = Table(title="Results by Configuration", show_header=True)
        table.add_column("Configuration", style="cyan")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Rate", justify="right")
        table.add_column("Cost", justify="right", style="yellow")

        for config_name, results in sorted(self.results_by_config.items()):
            if not results:
                continue
            passed = sum(1 for r in results if r.success)
            failed = len(results) - passed
            rate = (passed / len(results) * 100) if results else 0
            cost = sum(r.cost for r in results)

            rate_style = "green" if rate >= 75 else "yellow" if rate >= 50 else "red"
            table.add_row(
                config_name,
                str(passed),
                str(failed),
                f"[{rate_style}]{rate:.1f}%[/{rate_style}]",
                f"${cost:.4f}",
            )

        return table

    def render_recent_completions(self, n: int = 5) -> Panel:
        """Render panel showing recent completions."""
        recent = self.completed[-n:] if self.completed else []

        if not recent:
            content = Text("No completions yet", style="dim")
        else:
            lines = []
            for result in reversed(recent):
                status = (
                    Text("\u2713", style="green")
                    if result.success
                    else Text("\u2717", style="red")
                )
                line = Text.assemble(
                    ("  ", ""),
                    status,
                    (" ", ""),
                    (f"[{result.config_name}] ", "dim"),
                    (result.challenge_name, "white"),
                    (f" ({result.n_steps} steps)", "dim"),
                )
                lines.append(line)
            content = Group(*lines)

        return Panel(
            content,
            title="[bold]Recent Completions[/bold]",
            border_style="green" if self.completed else "dim",
        )

    def render_live_display(self) -> Group:
        """Render the full live display."""
        return Group(
            self.progress,
            "",
            self.render_active_runs(),
            "",
            self.render_recent_completions(),
        )

    def print_final_summary(self) -> None:
        """Print final summary after all benchmarks complete."""
        elapsed = (
            (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        )

        console.print()
        console.print("=" * 70)
        console.print("[bold]BENCHMARK COMPLETE[/bold]")
        console.print("=" * 70)
        console.print()

        # Summary table
        console.print(self.render_summary_table())

        # Overall stats
        total_passed = sum(1 for r in self.completed if r.success)
        total_failed = len(self.completed) - total_passed
        total_cost = sum(r.cost for r in self.completed)
        total_rate = (total_passed / len(self.completed) * 100) if self.completed else 0

        console.print()
        console.print(
            f"[bold]Total:[/bold] {total_passed}/{len(self.completed)} passed"
        )
        console.print(f"[bold]Success Rate:[/bold] {total_rate:.1f}%")
        console.print(f"[bold]Total Cost:[/bold] ${total_cost:.4f}")
        console.print(f"[bold]Elapsed Time:[/bold] {elapsed:.1f}s")
        console.print()


class QuietUI:
    """Minimal UI for quiet mode."""

    def __init__(self):
        self.completed: list[ChallengeResult] = []
        self.results_by_config: dict[str, list[ChallengeResult]] = {}

    def start(self, total_challenges: int, configs: list[str]) -> None:
        self.results_by_config = {config: [] for config in configs}
        console.print(f"Running {total_challenges} challenges...")

    def update(self, progress: ExecutionProgress) -> None:
        if progress.status in ("completed", "failed") and progress.result:
            self.completed.append(progress.result)
            self.results_by_config[progress.config_name].append(progress.result)
            status = "." if progress.result.success else "F"
            console.print(status, end="")

    def print_final_summary(self) -> None:
        console.print()
        for config_name, results in sorted(self.results_by_config.items()):
            if not results:
                continue
            passed = sum(1 for r in results if r.success)
            console.print(f"{config_name}: {passed}/{len(results)} passed")


class JsonUI:
    """JSON output mode for CI/scripting."""

    def __init__(self):
        self.completed: list[ChallengeResult] = []
        self.results_by_config: dict[str, list[ChallengeResult]] = {}

    def start(self, total_challenges: int, configs: list[str]) -> None:
        self.results_by_config = {config: [] for config in configs}

    def update(self, progress: ExecutionProgress) -> None:
        if progress.status in ("completed", "failed") and progress.result:
            self.completed.append(progress.result)
            self.results_by_config[progress.config_name].append(progress.result)

    def print_final_summary(self) -> None:
        import json

        output = {
            "results": {
                config: {
                    "passed": sum(1 for r in results if r.success),
                    "failed": sum(1 for r in results if not r.success),
                    "total": len(results),
                    "cost": sum(r.cost for r in results),
                    "challenges": [
                        {
                            "name": r.challenge_name,
                            "success": r.success,
                            "steps": r.n_steps,
                            "cost": r.cost,
                            "error": r.error_message,
                        }
                        for r in results
                    ],
                }
                for config, results in self.results_by_config.items()
            }
        }
        console.print_json(data=output)
