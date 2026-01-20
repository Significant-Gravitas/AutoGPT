"""CLI entry point for the direct benchmark harness."""

import os
import sys
from pathlib import Path
from typing import Optional

import click

from .challenge_loader import find_challenges_dir
from .harness import BenchmarkHarness
from .models import (
    MODEL_PRESETS,
    STRATEGIES,
    BenchmarkConfig,
    HarnessConfig,
    ModelConfig,
)
from .ui import console


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Direct Benchmark Harness for AutoGPT.

    Run benchmarks with direct agent instantiation - no HTTP server,
    parallel execution, faster results.
    """
    pass


@cli.command()
@click.option(
    "--strategies",
    "-s",
    default="one_shot",
    help=f"Comma-separated strategies to test. Available: {', '.join(STRATEGIES)}",
)
@click.option(
    "--models",
    "-m",
    default="claude",
    help=f"Comma-separated model presets. Available: {', '.join(MODEL_PRESETS.keys())}",
)
@click.option(
    "--categories",
    "-c",
    default=None,
    help="Filter by categories (comma-separated).",
)
@click.option(
    "--skip-category",
    "-S",
    "skip_categories",
    default=None,
    help="Exclude categories (comma-separated).",
)
@click.option(
    "--tests",
    "-t",
    default=None,
    help="Filter by test names (comma-separated).",
)
@click.option(
    "--attempts",
    "-N",
    default=1,
    type=int,
    help="Number of times to run each challenge.",
)
@click.option(
    "--parallel",
    "-p",
    default=4,
    type=int,
    help="Maximum parallel runs.",
)
@click.option(
    "--timeout",
    default=300,
    type=int,
    help="Per-challenge timeout in seconds.",
)
@click.option(
    "--cutoff",
    default=None,
    type=int,
    help="Override challenge time limit (seconds). Alias for --timeout.",
)
@click.option(
    "--no-cutoff",
    "--nc",
    "no_cutoff",
    is_flag=True,
    help="Disable the challenge time limit.",
)
@click.option(
    "--max-steps",
    default=50,
    type=int,
    help="Maximum steps per challenge.",
)
@click.option(
    "--maintain",
    is_flag=True,
    help="Run only regression tests (previously beaten challenges).",
)
@click.option(
    "--improve",
    is_flag=True,
    help="Run only non-regression tests (not consistently beaten).",
)
@click.option(
    "--explore",
    is_flag=True,
    help="Run only challenges that have never been beaten.",
)
@click.option(
    "--no-dep",
    is_flag=True,
    help="Run all selected challenges regardless of dependency success/failure.",
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Workspace root directory.",
)
@click.option(
    "--challenges-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to challenges directory.",
)
@click.option(
    "--reports-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to reports directory.",
)
@click.option(
    "--keep-answers",
    is_flag=True,
    help="Keep answer files for debugging.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Minimal output mode.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output with per-challenge details.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="JSON output mode for CI/scripting.",
)
@click.option(
    "--ci",
    "ci_mode",
    is_flag=True,
    help="CI mode: no live display, but shows completion blocks. Auto-enabled when CI env var is set.",
)
@click.option(
    "--fresh",
    is_flag=True,
    help="Clear all saved state and start fresh (don't resume).",
)
@click.option(
    "--retry-failures",
    is_flag=True,
    help="Re-run only the challenges that failed in the previous run.",
)
@click.option(
    "--reset-strategy",
    "reset_strategies",
    multiple=True,
    help="Reset saved results for specific strategy (can be used multiple times).",
)
@click.option(
    "--reset-model",
    "reset_models",
    multiple=True,
    help="Reset saved results for specific model (can be used multiple times).",
)
@click.option(
    "--reset-challenge",
    "reset_challenges",
    multiple=True,
    help="Reset saved results for specific challenge (can be used multiple times).",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output.",
)
def run(
    strategies: str,
    models: str,
    categories: Optional[str],
    skip_categories: Optional[str],
    tests: Optional[str],
    attempts: int,
    parallel: int,
    timeout: int,
    cutoff: Optional[int],
    no_cutoff: bool,
    max_steps: int,
    maintain: bool,
    improve: bool,
    explore: bool,
    no_dep: bool,
    workspace: Optional[Path],
    challenges_dir: Optional[Path],
    reports_dir: Optional[Path],
    keep_answers: bool,
    quiet: bool,
    verbose: bool,
    json_output: bool,
    ci_mode: bool,
    fresh: bool,
    retry_failures: bool,
    reset_strategies: tuple[str, ...],
    reset_models: tuple[str, ...],
    reset_challenges: tuple[str, ...],
    debug: bool,
):
    """Run benchmarks with specified configurations."""
    # Handle timeout/cutoff options
    if cutoff is not None:
        timeout = cutoff
    if no_cutoff:
        timeout = 0  # 0 means no timeout
    # Parse strategies
    strategy_list = [s.strip() for s in strategies.split(",")]
    invalid_strategies = [s for s in strategy_list if s not in STRATEGIES]
    if invalid_strategies:
        console.print(f"[red]Invalid strategies: {invalid_strategies}[/red]")
        console.print(f"Available: {STRATEGIES}")
        sys.exit(1)

    # Parse models
    model_list = [m.strip() for m in models.split(",")]
    invalid_models = [m for m in model_list if m not in MODEL_PRESETS]
    if invalid_models:
        console.print(f"[red]Invalid model presets: {invalid_models}[/red]")
        console.print(f"Available: {list(MODEL_PRESETS.keys())}")
        sys.exit(1)

    # Find challenges directory
    if challenges_dir is None:
        challenges_dir = find_challenges_dir()
        if challenges_dir is None:
            console.print(
                "[red]Could not find challenges directory. "
                "Please specify with --challenges-dir[/red]"
            )
            sys.exit(1)

    # Set up paths
    if workspace is None:
        workspace = Path.cwd() / ".benchmark_workspaces"

    if reports_dir is None:
        reports_dir = Path.cwd() / "reports"

    # Build configurations
    configs: list[BenchmarkConfig] = []
    for strategy in strategy_list:
        for model_name in model_list:
            model = MODEL_PRESETS[model_name]
            configs.append(
                BenchmarkConfig(
                    strategy=strategy,
                    model=model,
                    max_steps=max_steps,
                    timeout_seconds=timeout,
                )
            )

    # Create harness config
    harness_config = HarnessConfig(
        workspace_root=workspace,
        challenges_dir=challenges_dir,
        reports_dir=reports_dir,
        categories=categories.split(",") if categories else None,
        skip_categories=skip_categories.split(",") if skip_categories else None,
        test_names=tests.split(",") if tests else None,
        max_parallel=parallel,
        configs=configs,
        attempts=attempts,
        no_cutoff=no_cutoff,
        no_dep=no_dep,
        maintain=maintain,
        improve=improve,
        explore=explore,
        keep_answers=keep_answers,
        debug=debug,
        fresh=fresh,
        retry_failures=retry_failures,
        reset_strategies=list(reset_strategies) if reset_strategies else None,
        reset_models=list(reset_models) if reset_models else None,
        reset_challenges=list(reset_challenges) if reset_challenges else None,
    )

    # Determine UI mode
    # Auto-detect CI: CI env var set or not a TTY
    is_ci = ci_mode or os.environ.get("CI") == "true" or not sys.stdout.isatty()

    if json_output:
        ui_mode = "json"
    elif quiet:
        ui_mode = "quiet"
    elif is_ci:
        ui_mode = "ci"
    else:
        ui_mode = "default"

    # Print config summary (unless JSON mode)
    if ui_mode != "json":
        console.print()
        console.print("[bold]Direct Benchmark Harness[/bold]")
        console.print("=" * 50)
        console.print(f"Strategies: {strategy_list}")
        console.print(f"Models: {model_list}")
        console.print(f"Parallel: {parallel}")
        console.print(f"Challenges: {challenges_dir}")
        if categories:
            console.print(f"Categories: {categories}")
        if skip_categories:
            console.print(f"Skip Categories: {skip_categories}")
        if tests:
            console.print(f"Tests: {tests}")
        if attempts > 1:
            console.print(f"Attempts: {attempts}")
        if no_cutoff:
            console.print("Cutoff: [yellow]disabled[/yellow]")
        elif timeout != 300:
            console.print(f"Timeout: {timeout}s")
        if maintain:
            console.print("Mode: [cyan]maintain[/cyan] (regression tests only)")
        if improve:
            console.print("Mode: [cyan]improve[/cyan] (non-regression tests only)")
        if explore:
            console.print("Mode: [cyan]explore[/cyan] (never-beaten only)")
        if no_dep:
            console.print("Dependencies: [yellow]ignored[/yellow]")
        if keep_answers:
            console.print("Keep answers: [green]yes[/green]")
        if debug:
            console.print("Debug: [yellow]enabled[/yellow]")
        if ui_mode == "ci":
            console.print("UI Mode: [cyan]ci[/cyan] (no live display)")
        console.print("=" * 50)
        console.print()

    # Run harness
    harness = BenchmarkHarness(harness_config)
    results = harness.run_sync(ui_mode=ui_mode, verbose=verbose, debug=debug)

    # Exit with appropriate code
    if not results:
        sys.exit(1)

    total_passed = sum(sum(1 for r in res if r.success) for res in results.values())
    total_run = sum(len(res) for res in results.values())

    if total_passed == 0:
        sys.exit(1)
    elif total_passed < total_run:
        sys.exit(0)  # Some passed
    else:
        sys.exit(0)  # All passed


@cli.command()
@click.option(
    "--challenges-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to challenges directory.",
)
def list_challenges(challenges_dir: Optional[Path]):
    """List available challenges."""
    if challenges_dir is None:
        challenges_dir = find_challenges_dir()
        if challenges_dir is None:
            console.print(
                "[red]Could not find challenges directory. "
                "Please specify with --challenges-dir[/red]"
            )
            sys.exit(1)

    from .challenge_loader import ChallengeLoader

    loader = ChallengeLoader(challenges_dir)
    challenges = sorted(loader.load_all(), key=lambda c: c.name)

    console.print(f"\n[bold]Available Challenges ({len(challenges)})[/bold]\n")

    # Group by category
    by_category: dict[str, list[str]] = {}
    for c in challenges:
        for cat in c.category:
            if cat not in by_category:
                by_category[cat] = []
            if c.name not in by_category[cat]:
                by_category[cat].append(c.name)

    for cat in sorted(by_category.keys()):
        console.print(f"[cyan]{cat}[/cyan]: {', '.join(sorted(by_category[cat]))}")


@cli.command()
def list_models():
    """List available model presets."""
    console.print("\n[bold]Available Model Presets[/bold]\n")

    for name, config in sorted(MODEL_PRESETS.items()):
        console.print(f"[cyan]{name}[/cyan]:")
        console.print(f"  smart_llm: {config.smart_llm}")
        console.print(f"  fast_llm: {config.fast_llm}")
        if config.thinking_budget_tokens:
            console.print(f"  thinking_budget: {config.thinking_budget_tokens}")
        if config.reasoning_effort:
            console.print(f"  reasoning_effort: {config.reasoning_effort}")


@cli.command()
def list_strategies():
    """List available prompt strategies."""
    console.print("\n[bold]Available Strategies[/bold]\n")
    for s in STRATEGIES:
        console.print(f"  - {s}")


@cli.group()
def state():
    """Manage saved benchmark state (resume/reset)."""
    pass


@state.command("show")
@click.option(
    "--reports-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to reports directory.",
)
def state_show(reports_dir: Optional[Path]):
    """Show current benchmark state."""
    from .state import StateManager

    if reports_dir is None:
        reports_dir = Path.cwd() / "reports"

    state_manager = StateManager(reports_dir)
    summary = state_manager.get_summary()

    if summary["total_completed"] == 0:
        console.print("[dim]No saved state found.[/dim]")
        return

    console.print("\n[bold]Benchmark State[/bold]\n")
    console.print(f"Session ID: {summary['session_id']}")
    console.print(f"Started: {summary['started_at']}")
    console.print(f"Total completed: {summary['total_completed']}")
    console.print(f"  Passed: [green]{summary['passed']}[/green]")
    console.print(f"  Failed: [red]{summary['failed']}[/red]")
    console.print(f"Total cost: ${summary['total_cost']:.4f}")

    # Show unique strategies and models
    strategies = state_manager.list_strategies()
    models = state_manager.list_models()
    if strategies:
        console.print(f"\nStrategies: {', '.join(sorted(strategies))}")
    if models:
        console.print(f"Models: {', '.join(sorted(models))}")

    console.print(f"\nState file: {state_manager.state_file}")


@state.command("clear")
@click.option(
    "--reports-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to reports directory.",
)
@click.confirmation_option(prompt="Are you sure you want to clear all saved state?")
def state_clear(reports_dir: Optional[Path]):
    """Clear all saved benchmark state."""
    from .state import StateManager

    if reports_dir is None:
        reports_dir = Path.cwd() / "reports"

    state_manager = StateManager(reports_dir)
    prev_count = state_manager.get_completed_count()
    state_manager.reset()
    console.print(f"[green]Cleared {prev_count} completed runs.[/green]")


@state.command("reset")
@click.option(
    "--strategy",
    "-s",
    "strategies",
    multiple=True,
    help="Reset runs for specific strategy (can be used multiple times).",
)
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    help="Reset runs for specific model (can be used multiple times).",
)
@click.option(
    "--challenge",
    "-c",
    "challenges",
    multiple=True,
    help="Reset runs for specific challenge (can be used multiple times).",
)
@click.option(
    "--reports-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to reports directory.",
)
def state_reset(
    strategies: tuple[str, ...],
    models: tuple[str, ...],
    challenges: tuple[str, ...],
    reports_dir: Optional[Path],
):
    """Reset specific runs from saved state."""
    from .state import StateManager

    if not strategies and not models and not challenges:
        console.print(
            "[red]Must specify at least one of --strategy, --model, or --challenge[/red]"
        )
        sys.exit(1)

    if reports_dir is None:
        reports_dir = Path.cwd() / "reports"

    state_manager = StateManager(reports_dir)
    total_reset = 0

    for strat in strategies:
        count = state_manager.reset_matching(strategy=strat)
        total_reset += count
        if count > 0:
            console.print(f"Reset {count} runs for strategy: {strat}")

    for model in models:
        count = state_manager.reset_matching(model=model)
        total_reset += count
        if count > 0:
            console.print(f"Reset {count} runs for model: {model}")

    for chal in challenges:
        count = state_manager.reset_matching(challenge=chal)
        total_reset += count
        if count > 0:
            console.print(f"Reset {count} runs for challenge: {chal}")

    if total_reset == 0:
        console.print("[dim]No matching runs found.[/dim]")
    else:
        console.print(f"\n[green]Total reset: {total_reset} runs[/green]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
