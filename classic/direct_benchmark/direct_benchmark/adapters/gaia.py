"""GAIA benchmark adapter.

GAIA (General AI Assistant Benchmark) evaluates AI assistants on real-world tasks
requiring reasoning, tool use, and web browsing.

Dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
Leaderboard: https://huggingface.co/spaces/gaia-benchmark/leaderboard

Requires:
    - Hugging Face account with access to the gated dataset
    - HUGGING_FACE_HUB_TOKEN environment variable set
    - datasets and huggingface-hub packages
"""

import os
import re
import string
from pathlib import Path
from typing import Any, Iterator, Optional

from ..models import Challenge, ChallengeResult
from . import register_adapter
from .base import BenchmarkAdapter


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (GAIA-style normalization).

    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Collapse whitespace
    """
    # Lowercase
    answer = answer.lower()

    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)

    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))

    # Collapse whitespace
    answer = " ".join(answer.split())

    return answer.strip()


@register_adapter("gaia")
class GAIAAdapter(BenchmarkAdapter):
    """Adapter for the GAIA benchmark.

    GAIA provides real-world tasks at three difficulty levels:
        - Level 1: Simple tasks (single tool, straightforward reasoning)
        - Level 2: Moderate tasks (multiple tools, multi-step reasoning)
        - Level 3: Complex tasks (complex reasoning, tool chaining)

    Usage:
        adapter = GAIAAdapter(split="validation", subset="1")
        for challenge in adapter.load_challenges():
            # Run challenge...
    """

    name = "gaia"
    description = "GAIA - General AI Assistant Benchmark"

    HF_DATASET = "gaia-benchmark/GAIA"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        split: str = "validation",
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """Initialize the GAIA adapter.

        Args:
            cache_dir: Directory to cache the dataset.
            split: Dataset split - "validation" (has answers) or "test" (leaderboard).
            subset: Difficulty level filter - "1", "2", or "3".
            limit: Maximum number of challenges to load.
        """
        super().__init__(cache_dir, split, subset, limit)
        self._dataset = None
        self._file_cache: dict[str, Path] = {}

    def setup(self) -> None:
        """Download and cache the GAIA dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "GAIA adapter requires the 'datasets' package. "
                "Install with: pip install datasets huggingface-hub"
            )

        # Check for HF token
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "GAIA dataset requires authentication. "
                "Set HUGGING_FACE_HUB_TOKEN or HF_TOKEN environment variable. "
                "Get your token at https://huggingface.co/settings/tokens"
            )

        # Load dataset
        self._dataset = load_dataset(
            self.HF_DATASET,
            split=self.split,
            token=token,
            cache_dir=str(self.cache_dir / "gaia"),
        )

        # Download any associated files
        self._setup_file_cache()

        self._is_setup = True

    def _setup_file_cache(self) -> None:
        """Cache file attachments from the dataset."""
        if self._dataset is None:
            return

        file_dir = self.cache_dir / "gaia" / "files"
        file_dir.mkdir(parents=True, exist_ok=True)

        for item in self._dataset:
            if item.get("file_name") and item.get("file_path"):
                # The dataset includes file contents inline
                file_name = item["file_name"]
                self._file_cache[item["task_id"]] = file_dir / file_name

    def load_challenges(self) -> Iterator[Challenge]:
        """Load challenges from the GAIA dataset.

        Yields:
            Challenge objects for each GAIA task.
        """
        self.ensure_setup()

        if self._dataset is None:
            return

        count = 0
        for item in self._dataset:
            # Apply subset filter (difficulty level)
            if self.subset and str(item.get("Level")) != self.subset:
                continue

            # Apply limit
            if self.limit and count >= self.limit:
                break

            challenge = self._convert_to_challenge(item)
            yield challenge
            count += 1

    def _convert_to_challenge(self, item: dict[str, Any]) -> Challenge:
        """Convert a GAIA dataset item to a Challenge."""
        task_id = item["task_id"]
        question = item["Question"]
        level = item.get("Level", 1)
        final_answer = item.get("Final answer", "")
        file_name = item.get("file_name", "")

        # Build task description
        task = question
        if file_name:
            task = f"{question}\n\nA file has been provided: {file_name}"

        # Map GAIA levels to difficulty
        difficulty_map = {1: "easy", 2: "medium", 3: "hard"}
        difficulty = difficulty_map.get(level, "unknown")

        # Cutoff based on difficulty
        cutoff_map = {1: 180, 2: 300, 3: 600}
        cutoff = cutoff_map.get(level, 300)

        # Ground truth for evaluation
        ground_truth: dict[str, Any] = {
            "answer": final_answer,
            "eval": {"type": "gaia_match"},
        }

        # Create artifacts directory for any files
        artifacts_dir = self.cache_dir / "gaia" / "artifacts" / task_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Copy file if present
        if task_id in self._file_cache:
            src = self._file_cache[task_id]
            if src.exists():
                import shutil

                shutil.copy2(src, artifacts_dir / src.name)

        return Challenge(
            name=f"GAIA_{task_id}",
            task=task,
            category=["gaia", f"gaia_level_{level}"],
            difficulty=difficulty,
            cutoff=cutoff,
            ground_truth=ground_truth,
            artifacts_dir=artifacts_dir,
            source_path=artifacts_dir / "data.json",
        )

    def evaluate(
        self,
        result: ChallengeResult,
        challenge: Challenge,
        workspace_dir: Path,
    ) -> ChallengeResult:
        """Evaluate using GAIA-style normalized string matching.

        GAIA uses exact string matching after normalization:
        - Lowercase
        - Remove articles (a, an, the)
        - Remove punctuation
        - Collapse whitespace
        """
        ground = challenge.ground_truth
        expected = ground.get("answer", "")

        if not expected:
            # Test split has no answers - can't evaluate locally
            result.success = False
            result.score = 0.0
            result.error_message = (
                "No ground truth (test split - submit to leaderboard)"
            )
            return result

        # Get the agent's answer from output
        agent_answer = self._extract_answer(result)

        if not agent_answer:
            result.success = False
            result.score = 0.0
            result.error_message = "No answer found in agent output"
            return result

        # Normalize both answers
        normalized_expected = _normalize_answer(expected)
        normalized_actual = _normalize_answer(agent_answer)

        # Exact match after normalization
        if normalized_expected == normalized_actual:
            result.success = True
            result.score = 1.0
        else:
            result.success = False
            result.score = 0.0
            result.error_message = (
                f"Answer mismatch: expected '{expected}', got '{agent_answer}'"
            )

        return result

    def _extract_answer(self, result: ChallengeResult) -> str:
        """Extract the final answer from the agent's output.

        Looks for:
        1. Content in answer.txt file
        2. Final step result
        3. Any file with "answer" in the name
        """
        # Check for answer.txt
        for filename, content in result.output_files.items():
            if "answer" in filename.lower():
                return content.strip()

        # Check final step result
        if result.steps:
            last_step = result.steps[-1]
            if last_step.tool_name == "finish":
                # Try to extract answer from finish arguments
                reason = last_step.tool_args.get("reason", "")
                return reason.strip()

        return ""

    def get_challenge_count(self) -> Optional[int]:
        """Get the number of challenges in the dataset."""
        self.ensure_setup()
        if self._dataset is None:
            return None

        count = len(self._dataset)

        # Apply subset filter
        if self.subset:
            count = sum(
                1 for item in self._dataset if str(item.get("Level")) == self.subset
            )

        # Apply limit
        if self.limit:
            count = min(count, self.limit)

        return count

    def get_metadata(self) -> dict[str, Any]:
        """Get GAIA benchmark metadata."""
        metadata = super().get_metadata()
        metadata.update(
            {
                "dataset": self.HF_DATASET,
                "levels": ["1", "2", "3"],
                "splits": ["validation", "test"],
                "requires_auth": True,
                "leaderboard": (
                    "https://huggingface.co/spaces/gaia-benchmark/leaderboard"
                ),
            }
        )
        return metadata
