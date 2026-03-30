"""Base class for benchmark adapters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional

from ..models import Challenge, ChallengeResult


class BenchmarkAdapter(ABC):
    """Abstract base class for external benchmark adapters.

    Adapters translate external benchmark formats into the Challenge model
    used by the direct_benchmark harness.

    Subclasses must implement:
        - setup(): One-time initialization (download datasets, etc.)
        - load_challenges(): Yield Challenge objects from the benchmark
        - evaluate(): Custom evaluation logic for the benchmark

    Optionally override:
        - provision_environment(): Set up runtime environment for challenges
        - cleanup(): Clean up resources after benchmark run
    """

    # Override in subclasses
    name: str = "base"
    description: str = "Base benchmark adapter"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        split: str = "validation",
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """Initialize the adapter.

        Args:
            cache_dir: Directory to cache downloaded datasets.
            split: Dataset split to use (train/validation/test).
            subset: Optional subset filter (e.g., difficulty level, repo name).
            limit: Maximum number of challenges to load.
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "autogpt_benchmarks"
        self.split = split
        self.subset = subset
        self.limit = limit
        self._is_setup = False

    @abstractmethod
    def setup(self) -> None:
        """Perform one-time setup (download datasets, authenticate, etc.).

        This method is called before load_challenges() if not already setup.
        Should be idempotent - safe to call multiple times.
        """
        pass

    @abstractmethod
    def load_challenges(self) -> Iterator[Challenge]:
        """Load and yield challenges from the external benchmark.

        Yields:
            Challenge objects translated from the external format.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        result: ChallengeResult,
        challenge: Challenge,
        workspace_dir: Path,
    ) -> ChallengeResult:
        """Evaluate a challenge result using benchmark-specific logic.

        Args:
            result: The result from running the challenge.
            challenge: The challenge that was run.
            workspace_dir: Directory containing the agent's output.

        Returns:
            Updated ChallengeResult with success/score populated.
        """
        pass

    def provision_environment(self, challenge: Challenge) -> dict[str, Any]:
        """Set up runtime environment for a challenge.

        Override this for benchmarks that need Docker containers,
        database setup, etc.

        Args:
            challenge: The challenge to provision for.

        Returns:
            Environment configuration dict (passed to runner).
        """
        return {}

    def cleanup(self) -> None:
        """Clean up resources after benchmark run.

        Override this to stop containers, close connections, etc.
        """
        pass

    def ensure_setup(self) -> None:
        """Ensure setup() has been called."""
        if not self._is_setup:
            self.setup()
            self._is_setup = True

    def get_challenge_count(self) -> Optional[int]:
        """Get the total number of challenges without loading them.

        Returns:
            Number of challenges, or None if unknown without loading.
        """
        return None

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about this benchmark.

        Returns:
            Dict with benchmark metadata (name, description, splits, etc.).
        """
        return {
            "name": self.name,
            "description": self.description,
            "split": self.split,
            "subset": self.subset,
            "limit": self.limit,
        }
