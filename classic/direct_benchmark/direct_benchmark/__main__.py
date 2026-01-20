"""CLI entry point for the direct benchmark harness."""

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
    )

    # Determine UI mode
    if json_output:
        ui_mode = "json"
    elif quiet:
        ui_mode = "quiet"
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
        console.print("=" * 50)
        console.print()

    # Run harness
    harness = BenchmarkHarness(harness_config)
    results = harness.run_sync(ui_mode=ui_mode, verbose=verbose)

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


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
