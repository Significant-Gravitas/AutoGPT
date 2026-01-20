"""Rich UI components for the benchmark harness."""

from datetime import datetime
from typing import Optional

from rich.console import Console, Group, RenderableType
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

# Colors for different configs (cycling through for parallel runs)
CONFIG_COLORS = [
    "cyan",
    "green",
    "yellow",
    "magenta",
    "blue",
    "red",
    "bright_cyan",
    "bright_green",
]


class BenchmarkUI:
    """Rich UI for benchmark progress and results."""

    def __init__(self, max_parallel: int = 4, verbose: bool = False):
        self.max_parallel = max_parallel
        self.verbose = verbose
        self.start_time: Optional[datetime] = None

        # Track state
        self.active_runs: dict[str, str] = {}  # config_name -> challenge_name
        self.active_steps: dict[str, str] = {}  # config_name -> current step info
        self.completed: list[ChallengeResult] = []
        self.results_by_config: dict[str, list[ChallengeResult]] = {}
        self.recent_steps: list[tuple[str, str, int, str, bool]] = []  # Last N steps
        self.config_colors: dict[str, str] = {}  # config_name -> color

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
        # Assign colors to configs
        for i, config in enumerate(configs):
            self.config_colors[config] = CONFIG_COLORS[i % len(CONFIG_COLORS)]
        self.main_task = self.progress.add_task(
            "[cyan]Running benchmarks...", total=total_challenges
        )

    def get_config_color(self, config_name: str) -> str:
        """Get the assigned color for a config."""
        return self.config_colors.get(config_name, "white")

    def log_step(
        self,
        config_name: str,
        challenge_name: str,
        step_num: int,
        tool_name: str,
        result_preview: str,
        is_error: bool,
    ) -> None:
        """Log a step execution (called from AgentRunner)."""
        # Update active step info
        self.active_steps[config_name] = f"step {step_num}: {tool_name}"

        # Add to recent steps (keep last 10)
        self.recent_steps.append(
            (config_name, challenge_name, step_num, tool_name, is_error)
        )
        if len(self.recent_steps) > 10:
            self.recent_steps.pop(0)

        # In verbose mode, print immediately
        if self.verbose:
            color = self.get_config_color(config_name)
            status = "[red]ERR[/red]" if is_error else "[green]OK[/green]"
            console.print(
                f"[{color}][{config_name}][/{color}] {challenge_name} "
                f"step {step_num}: {tool_name} {status}"
            )

    def update(self, progress: ExecutionProgress) -> None:
        """Update UI with execution progress."""
        if progress.status == "starting":
            self.active_runs[progress.config_name] = progress.challenge_name
            self.active_steps[progress.config_name] = "starting..."
        elif progress.status in ("completed", "failed"):
            if progress.config_name in self.active_runs:
                del self.active_runs[progress.config_name]
            if progress.config_name in self.active_steps:
                del self.active_steps[progress.config_name]
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
            content = Text("Waiting for runs to start...", style="dim")
        else:
            lines = []
            for config_name, challenge_name in self.active_runs.items():
                color = self.get_config_color(config_name)
                step_info = self.active_steps.get(config_name, "")
                lines.append(
                    Text.assemble(
                        ("  ", ""),
                        ("\u25cf ", "yellow"),  # Bullet point
                        (f"{config_name}", color),
                        (" \u2192 ", "dim"),  # Arrow
                        (challenge_name, "white"),
                        (f" ({step_info})", "dim") if step_info else ("", ""),
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

    def render_recent_steps(self) -> Panel:
        """Render panel showing recent step executions."""
        if not self.recent_steps:
            content = Text("No steps yet", style="dim")
        else:
            lines = []
            for (
                config_name,
                challenge,
                step_num,
                tool_name,
                is_error,
            ) in self.recent_steps[-5:]:
                color = self.get_config_color(config_name)
                status = (
                    Text("\u2717", style="red")
                    if is_error
                    else Text("\u2713", style="green")
                )
                lines.append(
                    Text.assemble(
                        ("  ", ""),
                        status,
                        (" ", ""),
                        (f"[{config_name}]", color),
                        (" ", ""),
                        (f"{challenge} #{step_num}: ", "dim"),
                        (tool_name, "white"),
                    )
                )
            content = Group(*lines)

        return Panel(
            content,
            title="[bold]Recent Steps[/bold]",
            border_style="dim",
        )

    def render_live_display(self) -> Group:
        """Render the full live display."""
        return Group(
            self.progress,
            "",
            self.render_active_runs(),
            "",
            self.render_recent_steps(),
            "",
            self.render_recent_completions(),
        )

    def __rich_console__(self, console: Console, options) -> RenderableType:
        """Support for Rich Live display - called on each refresh."""
        yield self.render_live_display()

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
