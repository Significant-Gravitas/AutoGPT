"""State management for incremental benchmark runs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .models import ChallengeResult


class CompletedRun(BaseModel):
    """Record of a completed benchmark run."""

    config_name: str
    challenge_name: str
    attempt: int
    success: bool
    cost: float
    n_steps: int
    run_time_seconds: float
    error_message: Optional[str] = None
    completed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class BenchmarkState(BaseModel):
    """Persistent state for a benchmark session."""

    session_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_runs: dict[str, CompletedRun] = Field(default_factory=dict)

    # Track the original configuration to detect mismatches
    strategies: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    attempts: int = 1


class StateManager:
    """Manage persistent state for incremental benchmark runs.

    Tracks which config/challenge/attempt combinations have been completed,
    allowing the benchmark to resume from where it left off.
    """

    STATE_FILENAME = ".benchmark_state.json"

    def __init__(self, state_dir: Path):
        """Initialize the state manager.

        Args:
            state_dir: Directory to store the state file (usually reports_dir).
        """
        self.state_dir = state_dir
        self.state_file = state_dir / self.STATE_FILENAME
        self._state: Optional[BenchmarkState] = None

    @staticmethod
    def run_key(config_name: str, challenge_name: str, attempt: int = 1) -> str:
        """Generate a unique key for a run."""
        return f"{config_name}:{challenge_name}:{attempt}"

    def load(self) -> BenchmarkState:
        """Load state from disk, or create new state if none exists."""
        if self._state is not None:
            return self._state

        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self._state = BenchmarkState.model_validate(data)
            except Exception as e:
                print(f"Warning: Failed to load state file: {e}")
                print("Starting fresh session.")
                self._state = BenchmarkState()
        else:
            self._state = BenchmarkState()

        return self._state

    def save(self) -> None:
        """Save current state to disk."""
        if self._state is None:
            return

        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state.model_dump(), f, indent=2)

    def reset(self) -> None:
        """Clear all state and start fresh."""
        self._state = BenchmarkState()
        if self.state_file.exists():
            self.state_file.unlink()

    def is_completed(
        self, config_name: str, challenge_name: str, attempt: int = 1
    ) -> bool:
        """Check if a specific run has already been completed."""
        state = self.load()
        key = self.run_key(config_name, challenge_name, attempt)
        return key in state.completed_runs

    def get_completed_result(
        self, config_name: str, challenge_name: str, attempt: int = 1
    ) -> Optional[CompletedRun]:
        """Get the result of a completed run, if available."""
        state = self.load()
        key = self.run_key(config_name, challenge_name, attempt)
        return state.completed_runs.get(key)

    def mark_completed(self, result: ChallengeResult, attempt: int = 1) -> None:
        """Mark a run as completed and save immediately."""
        state = self.load()
        key = self.run_key(result.config_name, result.challenge_name, attempt)

        state.completed_runs[key] = CompletedRun(
            config_name=result.config_name,
            challenge_name=result.challenge_name,
            attempt=attempt,
            success=result.success,
            cost=result.cost,
            n_steps=result.n_steps,
            run_time_seconds=result.run_time_seconds,
            error_message=result.error_message,
        )

        # Save immediately for crash resilience
        self.save()

    def set_session_config(
        self, strategies: list[str], models: list[str], attempts: int
    ) -> None:
        """Record the session configuration for mismatch detection."""
        state = self.load()
        state.strategies = strategies
        state.models = models
        state.attempts = attempts
        self.save()

    def config_matches(
        self, strategies: list[str], models: list[str], attempts: int
    ) -> bool:
        """Check if the current config matches the saved session config.

        Returns True if no previous session exists or if configs match.
        """
        state = self.load()

        # No previous runs = matches (fresh session)
        if not state.completed_runs:
            return True

        # Check if configs match
        return (
            set(state.strategies) == set(strategies)
            and set(state.models) == set(models)
            and state.attempts == attempts
        )

    def get_summary(self) -> dict:
        """Get a summary of the current state."""
        state = self.load()
        completed = list(state.completed_runs.values())

        return {
            "session_id": state.session_id,
            "started_at": state.started_at,
            "total_completed": len(completed),
            "passed": sum(1 for r in completed if r.success),
            "failed": sum(1 for r in completed if not r.success),
            "total_cost": sum(r.cost for r in completed),
        }

    def get_completed_count(self) -> int:
        """Get the number of completed runs."""
        state = self.load()
        return len(state.completed_runs)

    def reset_matching(
        self,
        strategy: str | None = None,
        model: str | None = None,
        challenge: str | None = None,
    ) -> int:
        """Reset runs matching the given criteria.

        Args:
            strategy: Reset runs with this strategy (e.g., "reflexion").
            model: Reset runs with this model (e.g., "claude-thinking-25k").
            challenge: Reset runs for this challenge name.

        Returns:
            Number of runs reset.
        """
        state = self.load()
        keys_to_remove = []

        for key, run in state.completed_runs.items():
            # Parse config_name which is "{strategy}/{model}"
            parts = run.config_name.split("/")
            run_strategy = parts[0] if len(parts) > 0 else ""
            run_model = parts[1] if len(parts) > 1 else ""

            # Check if this run matches the criteria
            matches = True
            if strategy and run_strategy != strategy:
                matches = False
            if model and run_model != model:
                matches = False
            if challenge and run.challenge_name != challenge:
                matches = False

            if matches:
                keys_to_remove.append(key)

        # Remove matching runs
        for key in keys_to_remove:
            del state.completed_runs[key]

        if keys_to_remove:
            self.save()

        return len(keys_to_remove)

    def list_configs(self) -> set[str]:
        """List all unique config names in the state."""
        state = self.load()
        return {run.config_name for run in state.completed_runs.values()}

    def list_strategies(self) -> set[str]:
        """List all unique strategies in the state."""
        state = self.load()
        strategies = set()
        for run in state.completed_runs.values():
            parts = run.config_name.split("/")
            if parts:
                strategies.add(parts[0])
        return strategies

    def list_models(self) -> set[str]:
        """List all unique models in the state."""
        state = self.load()
        models = set()
        for run in state.completed_runs.values():
            parts = run.config_name.split("/")
            if len(parts) > 1:
                models.add(parts[1])
        return models

    def reset_failures(self) -> int:
        """Reset all failed runs so they can be retried.

        Returns:
            Number of runs reset.
        """
        state = self.load()
        keys_to_remove = [
            key for key, run in state.completed_runs.items() if not run.success
        ]

        for key in keys_to_remove:
            del state.completed_runs[key]

        if keys_to_remove:
            self.save()

        return len(keys_to_remove)

    def get_failure_count(self) -> int:
        """Get the number of failed runs."""
        state = self.load()
        return sum(1 for run in state.completed_runs.values() if not run.success)
