"""Main benchmark harness orchestrator."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from rich.live import Live

from .challenge_loader import ChallengeLoader
from .models import ChallengeResult, ExecutionProgress, HarnessConfig
from .parallel import ParallelExecutor
from .report import ReportGenerator
from .ui import BenchmarkUI, JsonUI, QuietUI, console


class BenchmarkHarness:
    """Main benchmark harness orchestrator."""

    def __init__(self, config: HarnessConfig):
        self.config = config
        self.loader = ChallengeLoader(config.challenges_dir)
        self.reporter = ReportGenerator(config.reports_dir)

    async def run(
        self,
        ui_mode: str = "default",
        verbose: bool = False,
    ) -> dict[str, list[ChallengeResult]]:
        """Run the full benchmark suite.

        Args:
            ui_mode: UI mode - "default" (rich), "quiet", or "json".
            verbose: Whether to show detailed per-challenge output.

        Returns:
            Dict mapping config_name -> list of ChallengeResult.
        """
        start_time = datetime.now()

        # Load challenges
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
            ui = BenchmarkUI(
                max_parallel=self.config.max_parallel,
                verbose=verbose,
            )

        # Initialize UI
        ui.start(total_runs, config_names)

        # Initialize results storage
        all_results: dict[str, list[ChallengeResult]] = {
            name: [] for name in config_names
        }

        # Create progress callback
        def on_progress(progress: ExecutionProgress) -> None:
            ui.update(progress)
            if progress.result:
                all_results[progress.config_name].append(progress.result)

        # Create step callback if UI supports it
        step_callback = None
        if hasattr(ui, "log_step"):
            step_callback = ui.log_step

        # Create executor
        executor = ParallelExecutor(
            max_parallel=self.config.max_parallel,
            on_progress=on_progress,
            on_step=step_callback,
            attempts=self.config.attempts,
            no_cutoff=self.config.no_cutoff,
        )

        # Ensure workspace exists
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)

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
            async for _ in executor.execute_matrix(
                self.config.configs,
                challenges,
                self.config.workspace_root,
            ):
                pass  # Results collected via callback

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
    ) -> dict[str, list[ChallengeResult]]:
        """Synchronous wrapper for run().

        Args:
            ui_mode: UI mode - "default" (rich), "quiet", or "json".
            verbose: Whether to show detailed per-challenge output.

        Returns:
            Dict mapping config_name -> list of ChallengeResult.
        """
        return asyncio.run(self.run(ui_mode=ui_mode, verbose=verbose))
