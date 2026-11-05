"""Tests for protect-dev-and-master.py.

Run: python3 .claude/hooks/test_protect_dev_and_master.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

HOOK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "protect-dev-and-master.py"
)


def git(directory: str, *args: str) -> None:
    subprocess.run(["git", "-C", directory, *args], check=True, capture_output=True)


def make_repo(directory: str, branch: str) -> None:
    git(directory, "init", "-q", "-b", branch)
    git(directory, "config", "user.email", "test@example.invalid")
    git(directory, "config", "user.name", "Test")
    with open(os.path.join(directory, "README.md"), "w") as file:
        file.write("hi\n")
    git(directory, "add", "README.md")
    git(directory, "commit", "-q", "-m", "init")


class ProtectDevAndMasterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repo = os.path.join(cls.tmp.name, "repo")
        cls.other = os.path.join(cls.tmp.name, "other")
        cls.outside = os.path.join(cls.tmp.name, "scratch")
        for directory in (cls.repo, cls.other, cls.outside):
            os.makedirs(directory)
        make_repo(cls.repo, "master")
        git(cls.repo, "branch", "dev")
        git(cls.repo, "branch", "john/secrt-1-fix")
        make_repo(cls.other, "master")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def checkout(self, branch: str) -> None:
        git(self.repo, "checkout", "-q", branch)

    def run_hook(self, payload, *args: str, cwd: str | None = None):
        env = {**os.environ, "CLAUDE_PROJECT_DIR": self.repo}
        stdin = (
            payload
            if isinstance(payload, str)
            else json.dumps({"cwd": cwd or self.repo, **payload})
        )
        return subprocess.run(
            [sys.executable, HOOK, *args],
            input=stdin,
            capture_output=True,
            text=True,
            env=env,
        )

    def bash(self, command: str, cwd: str | None = None):
        return self.run_hook(
            {"tool_name": "Bash", "tool_input": {"command": command}}, cwd=cwd
        )

    def assertBlocked(self, result) -> None:
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("AGENTS.md", result.stderr)

    def assertAllowed(self, result) -> None:
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_commits_on_dev_and_master_are_blocked(self) -> None:
        for branch in ("dev", "master"):
            self.checkout(branch)
            for command in (
                "git commit -m x",
                "git -c user.name=x commit -am x",
                "git merge john/secrt-1-fix",
                "git rebase origin/dev",
                "git cherry-pick abc123",
                "git revert HEAD",
                "git am patch.mbox",
                "GIT_EDITOR=true git commit --amend",
                "git add -A && git commit -m x",
            ):
                with self.subTest(branch=branch, command=command):
                    self.assertBlocked(self.bash(command))

    def test_commands_aimed_at_the_repo_from_elsewhere_are_blocked(self) -> None:
        self.checkout("dev")
        self.assertBlocked(
            self.bash(f"cd {self.repo} && git commit -m x", cwd=self.outside)
        )
        self.assertBlocked(
            self.bash(f"git -C {self.repo} commit -m x", cwd=self.outside)
        )

    def test_reading_and_leaving_dev_are_allowed(self) -> None:
        self.checkout("dev")
        for command in (
            "git status",
            "git log --oneline -5",
            "git diff",
            "git fetch origin dev",
            "git stash",
            "git checkout -b john/secrt-2-new origin/dev",
            "git switch -c john/secrt-2-new",
            "git merge --abort",
            "git rebase --abort",
            "ls -la",
        ):
            with self.subTest(command=command):
                self.assertAllowed(self.bash(command))

    def test_pushes_that_update_dev_or_master_are_blocked_from_any_branch(self) -> None:
        cases = {
            "dev": (
                "git push",
                "git push -u origin dev",
                "git push origin HEAD",
                "git push --force origin",
            ),
            "john/secrt-1-fix": (
                "git push origin dev",
                "git push origin master",
                "git push origin HEAD:dev",
                "git push origin john/secrt-1-fix:refs/heads/master",
                "git push origin +john/secrt-1-fix:dev",
                "git push origin :dev",
                "git push origin --delete dev",
                "git push --all origin",
                "git push --mirror origin",
                "git push -o ci.skip origin dev",
            ),
        }
        for branch, commands in cases.items():
            self.checkout(branch)
            for command in commands:
                with self.subTest(branch=branch, command=command):
                    self.assertBlocked(self.bash(command))

    def test_feature_branch_work_is_allowed(self) -> None:
        self.checkout("john/secrt-1-fix")
        for command in (
            "git commit -m x",
            "git push",
            "git push -u origin john/secrt-1-fix",
            "git push origin HEAD",
            "git push --force-with-lease origin john/secrt-1-fix",
            "git push origin dev-tools",
            "git push --tags",
            "git rebase origin/dev",
            "echo 'git push origin dev'",
            "git commit -F - <<'EOF'\nfix: x\n\n"
            "git push origin dev was the old way\nEOF",
        ):
            with self.subTest(command=command):
                self.assertAllowed(self.bash(command))

    def test_file_edits_are_blocked_only_on_dev_or_master_in_this_repo(self) -> None:
        in_repo = os.path.join(self.repo, "README.md")
        new_file = os.path.join(self.repo, "new", "dir", "file.py")
        for tool, key, path in (
            ("Edit", "file_path", in_repo),
            ("Write", "file_path", new_file),
            ("NotebookEdit", "notebook_path", os.path.join(self.repo, "nb.ipynb")),
        ):
            payload = {"tool_name": tool, "tool_input": {key: path}}
            self.checkout("dev")
            with self.subTest(tool=tool, branch="dev"):
                self.assertBlocked(self.run_hook(payload))
            self.checkout("john/secrt-1-fix")
            with self.subTest(tool=tool, branch="feature"):
                self.assertAllowed(self.run_hook(payload))
        self.checkout("dev")
        for path in (
            os.path.join(self.other, "README.md"),
            os.path.join(self.outside, "notes.md"),
        ):
            with self.subTest(path=path):
                self.assertAllowed(
                    self.run_hook(
                        {"tool_name": "Write", "tool_input": {"file_path": path}}
                    )
                )

    def test_github_mcp_writes_to_dev_or_master_are_blocked(self) -> None:
        for tool in (
            "mcp__github__push_files",
            "mcp__github__create_or_update_file",
            "mcp__github__delete_file",
        ):
            for branch, blocked in (
                ("dev", True),
                ("refs/heads/master", True),
                ("john/secrt-1-fix", False),
            ):
                result = self.run_hook(
                    {"tool_name": tool, "tool_input": {"branch": branch}}
                )
                with self.subTest(tool=tool, branch=branch):
                    if blocked:
                        self.assertBlocked(result)
                    else:
                        self.assertAllowed(result)

    def test_session_start_reminds_only_on_dev_or_master(self) -> None:
        self.checkout("dev")
        reminder = self.run_hook({}, "session-start")
        self.assertEqual(reminder.returncode, 0)
        self.assertIn("starts on `dev`", reminder.stdout)
        self.checkout("john/secrt-1-fix")
        quiet = self.run_hook({}, "session-start")
        self.assertEqual((quiet.returncode, quiet.stdout), (0, ""))

    def test_unreadable_input_is_let_through(self) -> None:
        self.assertAllowed(self.run_hook("not json"))
        self.assertAllowed(
            self.run_hook({"tool_name": "Read", "tool_input": {"file_path": "x"}})
        )


if __name__ == "__main__":
    unittest.main()
