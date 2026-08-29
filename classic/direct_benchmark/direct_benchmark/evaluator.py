"""Evaluate challenge results against ground truth."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .models import Challenge, ChallengeResult


class Evaluator:
    """Evaluate challenge results against ground truth."""

    def evaluate(
        self, result: ChallengeResult, challenge: Challenge
    ) -> ChallengeResult:
        """Evaluate a challenge result and update success/score.

        For timed-out challenges, we still run evaluation to populate the score
        (so we can show "would have passed"), but success remains False.
        """
        ground = challenge.ground_truth

        if not ground:
            # No ground truth defined, can't evaluate
            result.success = False
            result.score = 0.0
            result.error_message = (
                result.error_message or "No ground truth defined for evaluation"
            )
            return result

        # Get evaluation type
        eval_config = ground.get("eval", {})
        eval_type = eval_config.get("type", "file")

        # Get target files
        target_files = ground.get("files", [])

        # Collect content from output files
        content = self._collect_eval_content(result, target_files)

        # Run evaluation based on type
        try:
            if eval_type == "python":
                score = self._eval_python(result, challenge, target_files)
            elif eval_type == "pytest":
                score = self._eval_pytest(result, challenge)
            elif eval_type == "llm":
                # LLM evaluation not yet implemented, fall back to string match
                score = self._eval_string_match(content, ground)
            else:
                # Default: string matching (type "file" or unspecified)
                score = self._eval_string_match(content, ground)
        except Exception as e:
            score = 0.0
            result.error_message = f"Evaluation error: {e}"

        # Update result
        result.score = score
        # Timed-out challenges cannot pass, even if evaluation would succeed
        # (The score is still set so UI can show "would have passed")
        if result.timed_out:
            result.success = False
        else:
            result.success = score >= 0.9  # 90% threshold for success

        return result

    def _collect_eval_content(
        self,
        result: ChallengeResult,
        target_files: list[str],
    ) -> str:
        """Collect content from target files for evaluation."""
        contents: list[str] = []

        for pattern in target_files:
            for filepath, content in result.output_files.items():
                if self._matches_pattern(filepath, pattern):
                    contents.append(content)

        return "\n".join(contents)

    def _matches_pattern(self, filepath: str, pattern: str) -> bool:
        """Check if filepath matches the pattern."""
        # Extension pattern (e.g., ".txt")
        if pattern.startswith("."):
            return filepath.endswith(pattern)
        # Exact match or ends with pattern
        return filepath == pattern or filepath.endswith("/" + pattern)

    def _eval_string_match(self, content: str, ground: dict) -> float:
        """Evaluate using string contains/not contains."""
        should_contain = ground.get("should_contain", [])
        should_not_contain = ground.get("should_not_contain", [])
        case_sensitive = ground.get("case_sensitive", True)

        check_content = content if case_sensitive else content.lower()

        # Check should_contain
        for phrase in should_contain:
            check_phrase = phrase if case_sensitive else phrase.lower()
            if check_phrase not in check_content:
                return 0.0

        # Check should_not_contain
        for phrase in should_not_contain:
            check_phrase = phrase if case_sensitive else phrase.lower()
            if check_phrase in check_content:
                return 0.0

        # All checks passed
        return 1.0

    def _eval_python(
        self,
        result: ChallengeResult,
        challenge: Challenge,
        target_files: list[str],
    ) -> float:
        """Run Python test file and check output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Copy output files
            for filepath, content in result.output_files.items():
                dest = tmpdir_path / filepath
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)

            # Copy custom test file if exists
            custom_python = challenge.artifacts_dir / "custom_python"
            if custom_python.exists():
                for item in custom_python.iterdir():
                    if item.is_file():
                        shutil.copy2(item, tmpdir_path / item.name)

            # Run the test file(s)
            for target in target_files:
                test_file = tmpdir_path / target
                if test_file.exists() and test_file.suffix == ".py":
                    proc = subprocess.run(
                        [sys.executable, str(test_file)],
                        cwd=str(tmpdir_path),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if proc.returncode != 0:
                        return 0.0
                    if "error" in proc.stderr.lower():
                        return 0.0

        return 1.0

    def _eval_pytest(self, result: ChallengeResult, challenge: Challenge) -> float:
        """Run pytest and check if all tests pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Copy output files
            for filepath, content in result.output_files.items():
                dest = tmpdir_path / filepath
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)

            # Copy custom pytest files
            custom_python = challenge.artifacts_dir / "custom_python"
            if custom_python.exists():
                for item in custom_python.iterdir():
                    if item.is_file():
                        shutil.copy2(item, tmpdir_path / item.name)

            # Run pytest
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-v"],
                    cwd=str(tmpdir_path),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                return 1.0 if proc.returncode == 0 else 0.0
            except subprocess.TimeoutExpired:
                return 0.0
