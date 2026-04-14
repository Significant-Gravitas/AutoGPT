"""
Auto Git Commit Block — generates commit messages from diffs and commits changes.

Uses the LLM to analyze the git diff and produce a conventional commit message.
Supports: stage all, stage specific files, commit, push, and rollback operations.
"""

import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)


class GitOperation(str, Enum):
    STAGE_AND_COMMIT = "stage_and_commit"
    COMMIT_ONLY = "commit_only"
    PUSH = "push"
    ROLLBACK = "rollback"
    STATUS = "status"
    DIFF = "diff"
    LOG = "log"


def _run_git(args: list[str], cwd: str) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "Git command timed out."
    except FileNotFoundError:
        return 1, "", "Git not found. Ensure git is installed and in PATH."


def _generate_commit_message(diff: str, task_summary: str = "") -> str:
    """
    Generate a conventional commit message from a diff.
    This is a heuristic fallback; the LLM block should be used for better quality.
    """
    lines = diff.split("\n")
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    files_changed = set()
    for line in lines:
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) > 1:
                files_changed.add(parts[1].strip())

    # Determine commit type from file patterns
    commit_type = "feat"
    if any("test" in f.lower() for f in files_changed):
        commit_type = "test"
    elif any(f.endswith(".md") for f in files_changed):
        commit_type = "docs"
    elif any("fix" in task_summary.lower() or "bug" in task_summary.lower()):
        commit_type = "fix"

    scope = ""
    if files_changed:
        first_file = list(files_changed)[0]
        parts = first_file.split("/")
        if len(parts) > 1:
            scope = f"({parts[0]})"

    summary = task_summary.strip() if task_summary else f"update {len(files_changed)} file(s)"
    summary = summary[:72]  # Conventional commit subject line limit

    body = f"\n\nChanged {len(files_changed)} file(s): +{added} -{removed} lines."
    if task_summary:
        body += f"\n\nTask: {task_summary}"

    return f"{commit_type}{scope}: {summary}{body}"


class AutoGitCommitInput(BlockSchemaInput):
    operation: GitOperation = SchemaField(
        default=GitOperation.STAGE_AND_COMMIT,
        description="Git operation to perform.",
    )
    repo_path: str = SchemaField(
        default=".",
        description="Path to the git repository root.",
    )
    commit_message: str = SchemaField(
        default="",
        description=(
            "Commit message. If empty, auto-generates from diff. "
            "Pass the LLM-generated message here for best results."
        ),
    )
    task_summary: str = SchemaField(
        default="",
        description="Brief task summary used to generate commit message if none provided.",
    )
    files_to_stage: list = SchemaField(
        default_factory=list,
        description="Specific files to stage. If empty, stages all changes (git add -A).",
    )
    remote: str = SchemaField(
        default="origin",
        description="Remote name for push operation.",
    )
    branch: str = SchemaField(
        default="",
        description="Branch name for push. If empty, uses current branch.",
    )
    rollback_commits: int = SchemaField(
        default=1,
        description="Number of commits to roll back (for ROLLBACK operation).",
    )
    hard_reset: bool = SchemaField(
        default=False,
        description="If True, uses --hard reset (discards working tree changes). Use with caution.",
    )


class AutoGitCommitOutput(BlockSchemaOutput):
    success: bool = SchemaField(description="Whether the operation succeeded.")
    commit_hash: str = SchemaField(description="Git commit hash (if committed).")
    commit_message: str = SchemaField(description="The commit message used.")
    diff_summary: str = SchemaField(description="Summary of changes (files changed, insertions, deletions).")
    status: str = SchemaField(description="Git status output or operation result.")
    branch: str = SchemaField(description="Current branch name.")


class AutoGitCommitBlock(Block):
    """
    Automatically commits agent-made code changes to git with AI-generated commit messages.

    Stages changes, generates a conventional commit message from the diff (or uses
    a provided message), commits, and optionally pushes. Supports one-click rollback.
    """

    class Input(AutoGitCommitInput):
        pass

    class Output(AutoGitCommitOutput):
        pass

    def __init__(self):
        super().__init__(
            id="f6a7b8c9-d0e1-2345-fab0-678901234567",
            description=(
                "Auto-commits agent code changes with AI-generated conventional commit messages. "
                "Supports stage, commit, push, rollback, and status operations."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=AutoGitCommitBlock.Input,
            output_schema=AutoGitCommitBlock.Output,
            test_input={
                "operation": GitOperation.STATUS.value,
                "repo_path": ".",
            },
            test_output=[
                ("success", True),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        repo = input_data.repo_path

        # Get current branch
        _, branch, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)

        if input_data.operation == GitOperation.STATUS:
            code, out, err = _run_git(["status", "--short"], cwd=repo)
            yield "success", code == 0
            yield "commit_hash", ""
            yield "commit_message", ""
            yield "diff_summary", out
            yield "status", out or err
            yield "branch", branch

        elif input_data.operation == GitOperation.DIFF:
            code, out, err = _run_git(["diff", "--stat", "HEAD"], cwd=repo)
            yield "success", code == 0
            yield "commit_hash", ""
            yield "commit_message", ""
            yield "diff_summary", out
            yield "status", out or err
            yield "branch", branch

        elif input_data.operation == GitOperation.LOG:
            code, out, err = _run_git(
                ["log", "--oneline", "-20"], cwd=repo
            )
            yield "success", code == 0
            yield "commit_hash", ""
            yield "commit_message", ""
            yield "diff_summary", ""
            yield "status", out or err
            yield "branch", branch

        elif input_data.operation in (GitOperation.STAGE_AND_COMMIT, GitOperation.COMMIT_ONLY):
            # Stage files
            if input_data.operation == GitOperation.STAGE_AND_COMMIT:
                if input_data.files_to_stage:
                    for f in input_data.files_to_stage:
                        code, _, err = _run_git(["add", f], cwd=repo)
                        if code != 0:
                            yield "success", False
                            yield "commit_hash", ""
                            yield "commit_message", ""
                            yield "diff_summary", ""
                            yield "status", f"Failed to stage {f}: {err}"
                            yield "branch", branch
                            return
                else:
                    code, _, err = _run_git(["add", "-A"], cwd=repo)
                    if code != 0:
                        yield "success", False
                        yield "commit_hash", ""
                        yield "commit_message", ""
                        yield "diff_summary", ""
                        yield "status", f"Failed to stage changes: {err}"
                        yield "branch", branch
                        return

            # Get diff for message generation
            _, diff, _ = _run_git(["diff", "--cached", "--stat"], cwd=repo)
            _, full_diff, _ = _run_git(["diff", "--cached"], cwd=repo)

            # Generate or use provided commit message
            msg = input_data.commit_message.strip()
            if not msg:
                msg = _generate_commit_message(full_diff, input_data.task_summary)

            # Commit
            code, out, err = _run_git(["commit", "-m", msg], cwd=repo)
            if code != 0:
                yield "success", False
                yield "commit_hash", ""
                yield "commit_message", msg
                yield "diff_summary", diff
                yield "status", err or "Nothing to commit."
                yield "branch", branch
                return

            # Get commit hash
            _, commit_hash, _ = _run_git(["rev-parse", "--short", "HEAD"], cwd=repo)

            yield "success", True
            yield "commit_hash", commit_hash
            yield "commit_message", msg
            yield "diff_summary", diff
            yield "status", f"Committed {commit_hash}: {msg.split(chr(10))[0]}"
            yield "branch", branch

        elif input_data.operation == GitOperation.PUSH:
            push_branch = input_data.branch or branch
            code, out, err = _run_git(
                ["push", input_data.remote, push_branch], cwd=repo
            )
            yield "success", code == 0
            yield "commit_hash", ""
            yield "commit_message", ""
            yield "diff_summary", ""
            yield "status", out or err
            yield "branch", push_branch

        elif input_data.operation == GitOperation.ROLLBACK:
            reset_type = "--hard" if input_data.hard_reset else "--soft"
            target = f"HEAD~{input_data.rollback_commits}"
            code, out, err = _run_git(["reset", reset_type, target], cwd=repo)
            yield "success", code == 0
            yield "commit_hash", ""
            yield "commit_message", ""
            yield "diff_summary", ""
            yield "status", (
                f"Rolled back {input_data.rollback_commits} commit(s) "
                f"({'hard' if input_data.hard_reset else 'soft'} reset). {out or err}"
            )
            yield "branch", branch
