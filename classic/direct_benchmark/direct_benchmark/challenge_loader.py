"""Load challenges directly from data.json files."""

import json
from pathlib import Path
from typing import Iterator, Optional

from .models import Challenge


class ChallengeLoader:
    """Load challenges directly from the benchmark challenges directory."""

    def __init__(self, challenges_dir: Path, beaten_file: Optional[Path] = None):
        self.challenges_dir = challenges_dir
        self._beaten_challenges: Optional[dict[str, bool]] = None

        # Try to find the beaten file if not specified
        if beaten_file is None:
            # Look in common locations - prefer local direct_benchmark location
            pkg_dir = Path(__file__).parent.parent
            possible_paths = [
                pkg_dir / "challenges_already_beaten.json",  # direct_benchmark/
                challenges_dir.parent / "challenges_already_beaten.json",
                challenges_dir / "challenges_already_beaten.json",
            ]
            for path in possible_paths:
                if path.exists():
                    beaten_file = path
                    break

        self.beaten_file = beaten_file

    def _load_beaten_challenges(self) -> dict[str, bool]:
        """Load the challenges_already_beaten.json file."""
        if self._beaten_challenges is not None:
            return self._beaten_challenges

        self._beaten_challenges = {}
        if self.beaten_file and self.beaten_file.exists():
            with open(self.beaten_file) as f:
                data = json.load(f)
                # The file uses "Test{Name}" format, we need to strip the "Test" prefix
                for key, value in data.items():
                    name = key[4:] if key.startswith("Test") else key
                    self._beaten_challenges[name] = value

        return self._beaten_challenges

    def is_regression_test(self, challenge_name: str) -> bool:
        """Check if a challenge is a regression test (consistently beaten)."""
        beaten = self._load_beaten_challenges()
        return beaten.get(challenge_name, False)

    def has_been_passed(self, challenge_name: str) -> bool:
        """Check if a challenge has ever been passed."""
        beaten = self._load_beaten_challenges()
        return challenge_name in beaten

    def load_all(
        self,
        categories: Optional[list[str]] = None,
        skip_categories: Optional[list[str]] = None,
        names: Optional[list[str]] = None,
        maintain: bool = False,
        improve: bool = False,
        explore: bool = False,
    ) -> Iterator[Challenge]:
        """Load all challenges, optionally filtered by category or name.

        Args:
            categories: Only include challenges with at least one matching category.
            skip_categories: Exclude challenges with any matching category.
            names: Only include challenges with matching names.
            maintain: Only include regression tests (previously beaten consistently).
            improve: Only include non-regression tests (not consistently beaten).
            explore: Only include challenges never beaten.

        Yields:
            Challenge objects for each matching challenge.
        """
        for data_json in self.challenges_dir.rglob("data.json"):
            # Skip deprecated challenges
            if "deprecated" in str(data_json).lower():
                continue

            try:
                challenge = self._load_challenge(data_json)
            except Exception as e:
                # Skip malformed challenges
                print(f"Warning: Failed to load {data_json}: {e}")
                continue

            # Apply category filter (include)
            if categories:
                if not any(c in challenge.category for c in categories):
                    continue

            # Apply skip category filter (exclude)
            if skip_categories:
                if any(c in challenge.category for c in skip_categories):
                    continue

            # Apply name filter
            if names:
                if challenge.name not in names:
                    continue

            # Apply maintain/improve/explore filters
            if maintain or improve or explore:
                is_regression = self.is_regression_test(challenge.name)
                has_passed = self.has_been_passed(challenge.name)

                # --maintain: only challenges expected to pass (regression tests)
                if maintain and not is_regression:
                    continue

                # --improve: only challenges not consistently passed
                if improve and is_regression:
                    continue

                # --explore: only challenges never passed
                if explore and has_passed:
                    continue

            yield challenge

    def _load_challenge(self, data_json: Path) -> Challenge:
        """Load a single challenge from its data.json file."""
        with open(data_json) as f:
            data = json.load(f)

        # Extract difficulty from info block
        info = data.get("info", {})
        difficulty = info.get("difficulty", "unknown")

        return Challenge(
            name=data["name"],
            task=data["task"],
            category=data.get("category", []),
            difficulty=difficulty,
            cutoff=data.get("cutoff", 60),
            ground_truth=data.get("ground", {}),
            artifacts_dir=data_json.parent,
            source_path=data_json,
        )

    def list_categories(self) -> set[str]:
        """List all available categories."""
        categories: set[str] = set()
        for challenge in self.load_all():
            categories.update(challenge.category)
        return categories

    def list_challenges(self) -> list[str]:
        """List all available challenge names."""
        return sorted(c.name for c in self.load_all())


def find_challenges_dir() -> Optional[Path]:
    """Try to find the challenges directory automatically.

    Looks for common relative paths from the current working directory
    and the package location.
    """
    # First check relative to this file's location (preferred)
    pkg_dir = Path(__file__).parent.parent
    local_challenges = pkg_dir / "challenges"
    if local_challenges.exists() and (local_challenges / "abilities").exists():
        return local_challenges.resolve()

    # Fallback paths for backwards compatibility
    possible_paths = [
        Path("challenges"),
        Path("../challenges"),
        Path("../benchmark/agbenchmark/challenges"),
        Path("../../benchmark/agbenchmark/challenges"),
    ]

    for path in possible_paths:
        resolved = path.resolve()
        if resolved.exists() and (resolved / "abilities").exists():
            return resolved

    return None
