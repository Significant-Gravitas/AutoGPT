"""Parallel executor for benchmark runs."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Callable, Optional

from .evaluator import Evaluator
from .models import BenchmarkConfig, Challenge, ChallengeResult, ExecutionProgress
from .runner import AgentRunner, StepCallback

if TYPE_CHECKING:
    from .adapters.base import BenchmarkAdapter

# Type for skip predicate: (config_name, challenge_name, attempt) -> bool
SkipPredicate = Callable[[str, str, int], bool]


class ParallelExecutor:
    """Execute multiple benchmark configurations in parallel."""

    def __init__(
        self,
        max_parallel: int = 4,
        on_progress: Optional[Callable[[ExecutionProgress], None]] = None,
        on_step: Optional[StepCallback] = None,
        attempts: int = 1,
        no_cutoff: bool = False,
        skip_fn: Optional[SkipPredicate] = None,
        adapter: Optional["BenchmarkAdapter"] = None,
    ):
        self.max_parallel = max_parallel
        self.on_progress = on_progress
        self.on_step = on_step
        self.attempts = attempts
        self.no_cutoff = no_cutoff
        self.skip_fn = skip_fn
        self.adapter = adapter
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._evaluator = Evaluator()

    async def execute_matrix(
        self,
        configs: list[BenchmarkConfig],
        challenges: list[Challenge],
        workspace_root: Path,
    ) -> AsyncIterator[ChallengeResult]:
        """Execute all config/challenge combinations in parallel.

        Args:
            configs: List of benchmark configurations to test.
            challenges: List of challenges to run.
            workspace_root: Root directory for workspaces.

        Yields:
            ChallengeResult for each completed challenge.
        """
        # Create all tasks (including multiple attempts), skipping already-completed
        tasks = []
        skipped = 0
        for config in configs:
            for challenge in challenges:
                for attempt in range(1, self.attempts + 1):
                    # Check if this run should be skipped (already completed)
                    if self.skip_fn and self.skip_fn(
                        config.config_name, challenge.name, attempt
                    ):
                        skipped += 1
                        continue

                    task = asyncio.create_task(
                        self._run_with_semaphore(
                            config, challenge, workspace_root, attempt
                        )
                    )
                    tasks.append(task)

        # Store skipped count for external access
        self.skipped_count = skipped

        # Give all tasks a chance to start and acquire semaphore
        await asyncio.sleep(0)

        # Yield results as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result

    async def _run_with_semaphore(
        self,
        config: BenchmarkConfig,
        challenge: Challenge,
        workspace_root: Path,
        attempt: int = 1,
    ) -> ChallengeResult:
        """Run a single config/challenge combination with concurrency limit."""
        async with self._semaphore:
            # Yield to allow other tasks to acquire semaphore
            await asyncio.sleep(0)

            config_name = config.config_name
            challenge_display = (
                f"{challenge.name}"
                if self.attempts == 1
                else f"{challenge.name} (attempt {attempt})"
            )

            # Notify starting
            if self.on_progress:
                self.on_progress(
                    ExecutionProgress(
                        config_name=config_name,
                        challenge_name=challenge_display,
                        status="starting",
                    )
                )

            # Another yield to let UI update
            await asyncio.sleep(0)

            # Run the challenge (with modified timeout if no_cutoff is set)
            runner = AgentRunner(
                config,
                workspace_root,
                no_cutoff=self.no_cutoff,
                step_callback=self.on_step,
            )
            result = await runner.run_challenge(challenge, attempt=attempt)

            # Evaluate result - use adapter if available, otherwise standard evaluator
            if self.adapter is not None:
                result = self.adapter.evaluate(result, challenge, workspace_root)
            else:
                result = self._evaluator.evaluate(result, challenge)

            # Notify completion
            if self.on_progress:
                self.on_progress(
                    ExecutionProgress(
                        config_name=config_name,
                        challenge_name=challenge_display,
                        status="completed" if result.success else "failed",
                        result=result,
                    )
                )

            return result

    async def execute_sequential(
        self,
        configs: list[BenchmarkConfig],
        challenges: list[Challenge],
        workspace_root: Path,
    ) -> AsyncIterator[ChallengeResult]:
        """Execute all config/challenge combinations sequentially.

        Useful for debugging or when parallelism causes issues.

        Args:
            configs: List of benchmark configurations to test.
            challenges: List of challenges to run.
            workspace_root: Root directory for workspaces.

        Yields:
            ChallengeResult for each completed challenge.
        """
        for config in configs:
            for challenge in challenges:
                result = await self._run_with_semaphore(
                    config, challenge, workspace_root
                )
                yield result
