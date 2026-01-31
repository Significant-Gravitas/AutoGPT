"""SWE-bench adapter.

SWE-bench evaluates AI models on real-world GitHub issues from popular Python
repositories, requiring models to generate patches that fix the issues.

GitHub: https://github.com/SWE-bench/SWE-bench
Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench

Requires:
    - Docker Engine (for containerized evaluation)
    - swebench package (pip install swebench)
    - ~120GB disk space for full dataset
    - OR Modal for cloud-based evaluation
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional

from ..models import Challenge, ChallengeResult
from . import register_adapter
from .base import BenchmarkAdapter


@register_adapter("swe-bench")
class SWEBenchAdapter(BenchmarkAdapter):
    """Adapter for the SWE-bench benchmark.

    SWE-bench provides 2,294 real GitHub issues from 12 Python repositories.
    Models must generate patches that fix the issues, evaluated by running
    the repository's test suite.

    Subsets:
        - "full": All 2,294 instances
        - "lite": 300 curated instances
        - "verified": 500 human-validated solvable instances

    Usage:
        adapter = SWEBenchAdapter(subset="lite")
        for challenge in adapter.load_challenges():
            # Run challenge...
    """

    name = "swe-bench"
    description = "SWE-bench - Software Engineering Benchmark"

    HF_DATASET = "princeton-nlp/SWE-bench"
    HF_LITE = "princeton-nlp/SWE-bench_Lite"
    HF_VERIFIED = "princeton-nlp/SWE-bench_Verified"

    # Repository-specific timeout multipliers
    REPO_TIMEOUTS: dict[str, float] = {
        "django/django": 1.5,
        "matplotlib/matplotlib": 2.0,
        "scikit-learn/scikit-learn": 1.5,
        "sympy/sympy": 1.2,
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        split: str = "test",
        subset: Optional[str] = None,
        limit: Optional[int] = None,
        use_modal: bool = False,
    ):
        """Initialize the SWE-bench adapter.

        Args:
            cache_dir: Directory to cache the dataset.
            split: Dataset split - "dev" or "test".
            subset: Subset to use - "full", "lite", "verified", or a repo name.
            limit: Maximum number of challenges to load.
            use_modal: Use Modal for cloud-based evaluation instead of local Docker.
        """
        super().__init__(cache_dir, split, subset, limit)
        self._dataset = None
        self._use_modal = use_modal

    def setup(self) -> None:
        """Download and cache the SWE-bench dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SWE-bench adapter requires the 'datasets' package. "
                "Install with: pip install datasets"
            )

        # Select dataset based on subset
        if self.subset == "lite":
            dataset_name = self.HF_LITE
        elif self.subset == "verified":
            dataset_name = self.HF_VERIFIED
        else:
            dataset_name = self.HF_DATASET

        # Load dataset
        self._dataset = load_dataset(
            dataset_name,
            split=self.split,
            cache_dir=str(self.cache_dir / "swe_bench"),
        )

        # Check for Docker if not using Modal
        if not self._use_modal:
            self._check_docker()

        self._is_setup = True

    def _check_docker(self) -> None:
        """Verify Docker is available for evaluation."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not running")
        except FileNotFoundError:
            raise RuntimeError(
                "Docker is required for SWE-bench evaluation. "
                "Install Docker or use use_modal=True for cloud evaluation."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker is not responding")

    def load_challenges(self) -> Iterator[Challenge]:
        """Load challenges from the SWE-bench dataset.

        Yields:
            Challenge objects for each SWE-bench instance.
        """
        self.ensure_setup()

        if self._dataset is None:
            return

        count = 0
        for item in self._dataset:
            # Apply repo filter (if subset is a repo name)
            if self.subset and self.subset not in ("full", "lite", "verified"):
                if item.get("repo") != self.subset:
                    continue

            # Apply limit
            if self.limit and count >= self.limit:
                break

            challenge = self._convert_to_challenge(item)
            yield challenge
            count += 1

    def _convert_to_challenge(self, item: dict[str, Any]) -> Challenge:
        """Convert a SWE-bench dataset item to a Challenge."""
        instance_id = item["instance_id"]
        repo = item.get("repo", "unknown")
        problem_statement = item.get("problem_statement", "")
        base_commit = item.get("base_commit", "")
        hints_text = item.get("hints_text", "")

        # Build comprehensive task description
        task_parts = [
            f"Repository: {repo}",
            f"Base commit: {base_commit}",
            "",
            "Problem Statement:",
            problem_statement,
        ]

        if hints_text:
            task_parts.extend(["", "Hints:", hints_text])

        task_parts.extend(
            [
                "",
                "Your task: Generate a patch file (in unified diff format) that "
                "fixes the issue described above. The patch should be saved to "
                "'patch.diff' in your workspace.",
            ]
        )

        task = "\n".join(task_parts)

        # Determine difficulty based on repo complexity
        difficulty_map = {
            "astropy/astropy": "hard",
            "django/django": "medium",
            "flask/flask": "easy",
            "matplotlib/matplotlib": "hard",
            "pallets/flask": "easy",
            "psf/requests": "easy",
            "pydata/xarray": "medium",
            "pylint-dev/pylint": "medium",
            "pytest-dev/pytest": "medium",
            "scikit-learn/scikit-learn": "hard",
            "sphinx-doc/sphinx": "medium",
            "sympy/sympy": "hard",
        }
        difficulty = difficulty_map.get(repo, "medium")

        # Calculate timeout with repo-specific multiplier
        base_timeout = 600
        multiplier = self.REPO_TIMEOUTS.get(repo, 1.0)
        cutoff = int(base_timeout * multiplier)

        # Ground truth includes the gold patch for reference
        gold_patch = item.get("patch", "")
        test_patch = item.get("test_patch", "")

        ground_truth: dict[str, Any] = {
            "eval": {"type": "swe_bench"},
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "gold_patch": gold_patch,
            "test_patch": test_patch,
            "pass_to_pass": item.get("PASS_TO_PASS", ""),
            "fail_to_pass": item.get("FAIL_TO_PASS", ""),
        }

        # Create artifacts directory
        artifacts_dir = self.cache_dir / "swe_bench" / "artifacts" / instance_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save problem context for reference
        context_file = artifacts_dir / "context.json"
        with open(context_file, "w") as f:
            json.dump(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": base_commit,
                    "problem_statement": problem_statement,
                },
                f,
                indent=2,
            )

        return Challenge(
            name=f"SWE_{instance_id}",
            task=task,
            category=["swe-bench", f"swe-bench_{repo.replace('/', '_')}"],
            difficulty=difficulty,
            cutoff=cutoff,
            ground_truth=ground_truth,
            artifacts_dir=artifacts_dir,
            source_path=context_file,
        )

    def evaluate(
        self,
        result: ChallengeResult,
        challenge: Challenge,
        workspace_dir: Path,
    ) -> ChallengeResult:
        """Evaluate using SWE-bench's Docker-based test harness.

        The agent's patch is applied to the repository in a Docker container,
        and the test suite is run to verify the fix.
        """
        ground = challenge.ground_truth

        # Get the generated patch
        patch_content = self._extract_patch(result)

        if not patch_content:
            result.success = False
            result.score = 0.0
            result.error_message = "No patch.diff found in agent output"
            return result

        # Run evaluation
        if self._use_modal:
            eval_result = self._evaluate_with_modal(ground, patch_content)
        else:
            eval_result = self._evaluate_with_docker(ground, patch_content)

        result.success = eval_result["success"]
        result.score = eval_result["score"]
        if eval_result.get("error"):
            result.error_message = eval_result["error"]

        return result

    def _extract_patch(self, result: ChallengeResult) -> str:
        """Extract the patch from the agent's output."""
        # Look for patch.diff file
        for filename, content in result.output_files.items():
            if filename.endswith("patch.diff") or filename.endswith(".patch"):
                return content

        # Look for diff content in any output file
        for filename, content in result.output_files.items():
            if content.strip().startswith("diff --git") or content.strip().startswith(
                "---"
            ):
                return content

        return ""

    def _evaluate_with_docker(
        self, ground: dict[str, Any], patch: str
    ) -> dict[str, Any]:
        """Run evaluation using local Docker."""
        try:
            # Try to import swebench harness
            from swebench.harness.run_evaluation import run_evaluation
        except ImportError:
            return {
                "success": False,
                "score": 0.0,
                "error": (
                    "swebench package not installed. "
                    "Install with: pip install swebench"
                ),
            }

        instance_id = ground["instance_id"]
        # These are available for future use in more sophisticated evaluation
        _repo = ground["repo"]  # noqa: F841
        _base_commit = ground["base_commit"]  # noqa: F841

        # Write patch to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch)
            patch_file = f.name

        # Initialize predictions_file path before try block
        predictions_file = Path(patch_file).with_suffix(".json")

        try:
            # Create predictions file for swebench
            predictions = [
                {
                    "instance_id": instance_id,
                    "model_name_or_path": "autogpt",
                    "model_patch": patch,
                }
            ]

            with open(predictions_file, "w") as f:
                json.dump(predictions, f)

            # Run evaluation
            results = run_evaluation(
                predictions_path=str(predictions_file),
                swe_bench_tasks=self.HF_DATASET,
                log_dir=str(self.cache_dir / "swe_bench" / "logs"),
                testbed=str(self.cache_dir / "swe_bench" / "testbed"),
                skip_existing=False,
                timeout=1800,
                verbose=False,
            )

            # Check results
            if instance_id in results:
                instance_result = results[instance_id]
                resolved = instance_result.get("resolved", False)
                return {
                    "success": resolved,
                    "score": 1.0 if resolved else 0.0,
                    "error": None if resolved else "Tests did not pass",
                }
            else:
                return {
                    "success": False,
                    "score": 0.0,
                    "error": "Evaluation did not produce results",
                }

        except Exception as e:
            return {
                "success": False,
                "score": 0.0,
                "error": f"Evaluation failed: {str(e)}",
            }

        finally:
            # Cleanup temp files
            Path(patch_file).unlink(missing_ok=True)
            predictions_file.unlink(missing_ok=True)

    def _evaluate_with_modal(
        self, ground: dict[str, Any], patch: str
    ) -> dict[str, Any]:
        """Run evaluation using Modal cloud infrastructure."""
        try:
            import modal  # noqa: F401
        except ImportError:
            return {
                "success": False,
                "score": 0.0,
                "error": (
                    "Modal package not installed. " "Install with: pip install modal"
                ),
            }

        # Modal evaluation requires environment setup
        # This is a simplified interface - full implementation would use
        # modal's SWE-bench harness
        return {
            "success": False,
            "score": 0.0,
            "error": (
                "Modal evaluation not yet implemented. "
                "Use local Docker evaluation or submit to SWE-bench leaderboard."
            ),
        }

    def provision_environment(self, challenge: Challenge) -> dict[str, Any]:
        """Provide repository context for the challenge."""
        ground = challenge.ground_truth
        return {
            "repo": ground.get("repo"),
            "base_commit": ground.get("base_commit"),
            "clone_url": f"https://github.com/{ground.get('repo')}.git",
        }

    def get_challenge_count(self) -> Optional[int]:
        """Get the number of challenges in the dataset."""
        self.ensure_setup()
        if self._dataset is None:
            return None

        count = len(self._dataset)

        # Apply repo filter
        if self.subset and self.subset not in ("full", "lite", "verified"):
            count = sum(1 for item in self._dataset if item.get("repo") == self.subset)

        # Apply limit
        if self.limit:
            count = min(count, self.limit)

        return count

    def get_metadata(self) -> dict[str, Any]:
        """Get SWE-bench metadata."""
        metadata = super().get_metadata()
        metadata.update(
            {
                "datasets": {
                    "full": self.HF_DATASET,
                    "lite": self.HF_LITE,
                    "verified": self.HF_VERIFIED,
                },
                "subsets": ["full", "lite", "verified"]
                + list(self.REPO_TIMEOUTS.keys()),
                "splits": ["dev", "test"],
                "requires_docker": not self._use_modal,
                "leaderboard": "https://www.swebench.com/",
            }
        )
        return metadata
