"""Main benchmark harness orchestrator."""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from rich.live import Live

from .challenge_loader import ChallengeLoader
from .models import ChallengeResult, ExecutionProgress, HarnessConfig
from .parallel import ParallelExecutor
from .report import ReportGenerator
from .state import StateManager
from .ui import BenchmarkUI, JsonUI, QuietUI, console

if TYPE_CHECKING:
    from .adapters.base import BenchmarkAdapter


class BenchmarkHarness:
    """Main benchmark harness orchestrator."""

    def __init__(self, config: HarnessConfig):
        self.config = config
        self.reporter = ReportGenerator(config.reports_dir)
        self.state_manager = StateManager(config.reports_dir)

        # Initialize challenge source (adapter or loader)
        self.adapter: Optional["BenchmarkAdapter"] = None
        self.loader: Optional[ChallengeLoader] = None

        if config.external_benchmark:
            self._init_adapter()
        else:
            self.loader = ChallengeLoader(config.challenges_dir)

    def _init_adapter(self) -> None:
        """Initialize external benchmark adapter."""
        from .adapters import get_adapter

        assert self.config.external_benchmark is not None
        adapter_cls = get_adapter(self.config.external_benchmark)
        if adapter_cls is None:
            from .adapters import list_adapters

            available = list_adapters()
            raise ValueError(
                f"Unknown benchmark: {self.config.external_benchmark}. "
                f"Available: {available}"
            )

        # Determine cache directory
        cache_dir = self.config.benchmark_cache_dir
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "autogpt_benchmarks"

        # Create adapter instance
        self.adapter = adapter_cls(
            cache_dir=cache_dir,
            split=self.config.benchmark_split,
            subset=self.config.benchmark_subset,
            limit=self.config.benchmark_limit,
        )

    async def run(
        self,
        ui_mode: str = "default",
        verbose: bool = False,
        debug: bool = False,
    ) -> dict[str, list[ChallengeResult]]:
        """Run the full benchmark suite.

        Args:
            ui_mode: UI mode - "default" (rich Live display), "ci" (no Live,
                shows completion blocks, auto-enabled in CI environments),
                "quiet", or "json".
            verbose: Whether to show detailed per-challenge output.
            debug: Whether to enable debug mode.

        Returns:
            Dict mapping config_name -> list of ChallengeResult.
        """
        start_time = datetime.now()

        # Handle state management (resume/fresh)
        strategy_names = list({c.strategy for c in self.config.configs})
        model_names = list({c.model.name for c in self.config.configs})

        if self.config.fresh:
            self.state_manager.reset()
            if ui_mode != "json":
                console.print(
                    "[yellow]Starting fresh (cleared previous state)[/yellow]"
                )
        elif self.config.retry_failures:
            failure_count = self.state_manager.get_failure_count()
            if failure_count > 0:
                self.state_manager.reset_failures()
                if ui_mode != "json":
                    console.print(
                        f"[yellow]Retrying {failure_count} failed runs...[/yellow]"
                    )
            else:
                if ui_mode != "json":
                    console.print("[dim]No failures to retry.[/dim]")
        elif (
            self.config.reset_strategies
            or self.config.reset_models
            or self.config.reset_challenges
        ):
            # Handle selective resets
            total_reset = 0
            if self.config.reset_strategies:
                for strat in self.config.reset_strategies:
                    count = self.state_manager.reset_matching(strategy=strat)
                    total_reset += count
                    if ui_mode != "json" and count > 0:
                        console.print(
                            f"[yellow]Reset {count} runs for strategy: {strat}[/yellow]"
                        )
            if self.config.reset_models:
                for model in self.config.reset_models:
                    count = self.state_manager.reset_matching(model=model)
                    total_reset += count
                    if ui_mode != "json" and count > 0:
                        console.print(
                            f"[yellow]Reset {count} runs for model: {model}[/yellow]"
                        )
            if self.config.reset_challenges:
                for chal in self.config.reset_challenges:
                    count = self.state_manager.reset_matching(challenge=chal)
                    total_reset += count
                    if ui_mode != "json" and count > 0:
                        console.print(
                            f"[yellow]Reset {count} runs for challenge: {chal}[/yellow]"
                        )
        else:
            # Check for config mismatch
            if not self.state_manager.config_matches(
                strategy_names, model_names, self.config.attempts
            ):
                prev_completed = self.state_manager.get_completed_count()
                if prev_completed > 0:
                    msg = (
                        f"[yellow]Warning: Config changed ({prev_completed} "
                        f"completed). Use --fresh to start over.[/yellow]"
                    )
                    console.print(msg)
                    self.state_manager.reset()

        # Save current config for future mismatch detection
        self.state_manager.set_session_config(
            strategy_names, model_names, self.config.attempts
        )

        # Load challenges (from adapter or local loader)
        if self.adapter:
            # External benchmark - load via adapter
            if ui_mode != "json":
                subset_str = (
                    f", subset={self.config.benchmark_subset}"
                    if self.config.benchmark_subset
                    else ""
                )
                limit_str = (
                    f", limit={self.config.benchmark_limit}"
                    if self.config.benchmark_limit
                    else ""
                )
                console.print(
                    f"[cyan]Loading {self.config.external_benchmark} benchmark "
                    f"(split={self.config.benchmark_split}{subset_str}{limit_str})"
                    f"...[/cyan]"
                )
            assert self.adapter is not None
            challenges = list(self.adapter.load_challenges())
        else:
            # Local challenges - load via ChallengeLoader
            assert self.loader is not None
            challenges = list(
                self.loader.load_all(
                    categories=self.config.categories,
                    skip_categories=self.config.skip_categories,
                    names=self.config.test_names,
                    maintain=self.config.maintain,
                    improve=self.config.improve,
                    explore=self.config.explore,
                )
            )

        if not challenges:
            console.print("[red]No challenges found matching filters[/red]")
            return {}

        if not self.config.configs:
            console.print("[red]No configurations specified[/red]")
            return {}

        # Calculate total runs (including multiple attempts)
        total_runs = len(challenges) * len(self.config.configs) * self.config.attempts
        config_names = [c.config_name for c in self.config.configs]

        # Create UI
        ui: Union[BenchmarkUI, QuietUI, JsonUI]
        if ui_mode == "quiet":
            ui = QuietUI()
        elif ui_mode == "json":
            ui = JsonUI()
        else:
            # "default" or "ci" mode - both use BenchmarkUI
            # ci mode just skips the Live display
            ui = BenchmarkUI(
                max_parallel=self.config.max_parallel,
                verbose=verbose,
                debug=debug,
            )

        # Initialize UI
        ui.start(total_runs, config_names)

        # Initialize results storage
        all_results: dict[str, list[ChallengeResult]] = {
            name: [] for name in config_names
        }

        # Helper to parse challenge name and attempt from display name
        def parse_challenge_display(display_name: str) -> tuple[str, int]:
            """Parse 'ChallengeName (attempt N)' -> ('ChallengeName', N)"""
            match = re.match(r"^(.+?) \(attempt (\d+)\)$", display_name)
            if match:
                return match.group(1), int(match.group(2))
            return display_name, 1

        # Create progress callback that also saves state
        def on_progress(progress: ExecutionProgress) -> None:
            ui.update(progress)
            if progress.result:
                all_results[progress.config_name].append(progress.result)
                # Mark as completed in state manager
                _, attempt = parse_challenge_display(progress.challenge_name)
                self.state_manager.mark_completed(progress.result, attempt)

        # Create step callback if UI supports it
        step_callback = getattr(ui, "log_step", None)

        # Create skip function for resume functionality
        def should_skip(config_name: str, challenge_name: str, attempt: int) -> bool:
            return self.state_manager.is_completed(config_name, challenge_name, attempt)

        # Create executor
        executor = ParallelExecutor(
            max_parallel=self.config.max_parallel,
            on_progress=on_progress,
            on_step=step_callback,
            attempts=self.config.attempts,
            no_cutoff=self.config.no_cutoff,
            skip_fn=should_skip,
            adapter=self.adapter,
        )

        # Ensure workspace exists
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)

        # Check how many will be skipped (already completed)
        previously_completed = self.state_manager.get_completed_count()
        if previously_completed > 0 and ui_mode not in ("json", "quiet"):
            console.print(
                f"[cyan]Resuming: {previously_completed} runs already completed, "
                f"will run remaining...[/cyan]"
            )
            console.print()

        # Run with or without live display
        if isinstance(ui, BenchmarkUI) and ui_mode == "default":
            # Pass the UI object itself so Live can refresh it
            with Live(ui, console=console, refresh_per_second=4):
                async for _ in executor.execute_matrix(
                    self.config.configs,
                    challenges,
                    self.config.workspace_root,
                ):
                    pass  # Results collected via callback
        else:
            # CI mode or non-BenchmarkUI - no Live display
            if ui_mode == "ci" and isinstance(ui, BenchmarkUI):
                remaining = total_runs - previously_completed
                console.print(
                    f"[bold]Running {remaining} benchmark runs "
                    f"(parallel={self.config.max_parallel})...[/bold]"
                )
                console.print()

            completed_count = 0
            async for _ in executor.execute_matrix(
                self.config.configs,
                challenges,
                self.config.workspace_root,
            ):
                completed_count += 1
                # For CI mode, print progress every 10 completions
                remaining = total_runs - previously_completed
                if ui_mode == "ci" and completed_count % 10 == 0:
                    console.print(
                        f"[dim]Progress: {completed_count}/{remaining} completed[/dim]"
                    )

        end_time = datetime.now()

        # Generate reports
        for config_name, results in all_results.items():
            if results:
                self.reporter.generate_report(
                    results, config_name, start_time, end_time
                )

        # Generate comparison report
        if len(all_results) > 1:
            self.reporter.generate_comparison_report(all_results, end_time)

        # Print final summary
        ui.print_final_summary()

        return all_results

    def run_sync(
        self,
        ui_mode: str = "default",
        verbose: bool = False,
        debug: bool = False,
    ) -> dict[str, list[ChallengeResult]]:
        """Synchronous wrapper for run().

        Args:
            ui_mode: UI mode - "default" (rich Live), "ci" (no Live, completion
                blocks only), "quiet", or "json".
            verbose: Whether to show detailed per-challenge output.
            debug: Whether to enable debug mode (shows all logs).

        Returns:
            Dict mapping config_name -> list of ChallengeResult.
        """
        return asyncio.run(self.run(ui_mode=ui_mode, verbose=verbose, debug=debug))
