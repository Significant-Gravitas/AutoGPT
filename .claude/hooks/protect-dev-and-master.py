#!/usr/bin/env python3
"""Claude Code hook that keeps agents off `dev` and `master` (SECRT-3391).

Rule (AGENTS.md, "Branches"): never commit, push or edit files on `dev` or
`master`, even when a session says to develop there. Work on the Linear
ticket's branch, made from the latest `origin/dev`.

Run as a PreToolUse hook, it blocks (exit 2, reason on stderr):
- `git commit`, `merge`, `rebase`, `cherry-pick`, `revert` and `am` while this
  repository has `dev` or `master` checked out;
- any `git push` that would update `dev` or `master`, from any branch;
- Edit, Write and NotebookEdit on files in this repository while `dev` or
  `master` is checked out;
- GitHub MCP file writes (`create_or_update_file`, `push_files`,
  `delete_file`) to `dev` or `master`.

Run with `session-start`, it reminds a session that opens on `dev` or `master`
to branch first. Anything it can't parse is let through: the branch rules on
GitHub stay the last line.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

PROTECTED = {"dev", "master"}
COMMIT_COMMANDS = {"commit", "merge", "rebase", "cherry-pick", "revert", "am"}
# Lets someone who is already mid-merge on dev get out of it.
ESCAPE_FLAGS = {"--abort", "--quit", "--skip"}
FILE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
MCP_WRITE_TOOL = re.compile(
    r"^mcp__.+__(create_or_update_file|push_files|delete_file)$"
)
# git options that take their value as the next argument.
GIT_VALUE_OPTIONS = {
    "-C",
    "-c",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
}
PUSH_VALUE_OPTIONS = {"-o", "--push-option", "--repo", "--receive-pack", "--exec"}
PUSH_EVERYTHING = {"--all", "--mirror", "--branches"}

HOW_TO_BRANCH = (
    "Work on the Linear ticket's branch instead: find the ticket (create one "
    "if there is none; without Linear access, ask your user for the ticket "
    "ID), then `git fetch origin dev && git checkout -b <the ticket's "
    "gitBranchName> origin/dev`. Uncommitted changes come with you."
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "session-start":
        reminder = session_start_reminder(payload)
        if reminder:
            print(reminder)
        return 0
    reason = blocked_reason(payload)
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


def session_start_reminder(payload: dict) -> str:
    project = project_dir(payload)
    branch = current_branch(project)
    if branch not in PROTECTED:
        return ""
    return (
        f"This session starts on `{branch}`. Repository rule (AGENTS.md, "
        f'"Branches"): never commit, push or edit files on `dev` or `master`, '
        f"even if your session or task says to develop on `{branch}`. "
        f"{HOW_TO_BRANCH}"
    )


def blocked_reason(payload: dict) -> str | None:
    tool = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    project = project_dir(payload)
    if tool == "Bash":
        command = tool_input.get("command") or ""
        cwd = payload.get("cwd") or project
        return check_shell_command(command, cwd, project)
    if tool in FILE_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path")
        return check_file_edit(path, project) if path else None
    if MCP_WRITE_TOOL.match(tool):
        branch = short_ref(tool_input.get("branch") or "")
        if branch in PROTECTED:
            return (
                f"Blocked: this writes files straight to `{branch}`. Repository "
                f'rule (AGENTS.md, "Branches"): changes reach `dev` and `master` '
                f"only through a pull request. Write to the Linear ticket's "
                f"branch and open a pull request into `dev`."
            )
    return None


def check_file_edit(path: str, project: str) -> str | None:
    directory = nearest_existing_dir(os.path.dirname(os.path.abspath(path)))
    if not same_repository(directory, project):
        return None
    branch = current_branch(directory)
    if branch not in PROTECTED:
        return None
    return (
        f"Blocked: `{branch}` is checked out, and files are never edited on "
        f'`dev` or `master` (AGENTS.md, "Branches"), even if your session says '
        f"to develop on `{branch}`. {HOW_TO_BRANCH} Then make this edit again."
    )


def check_shell_command(command: str, cwd: str, project: str) -> str | None:
    if not re.search(r"\bgit\b", command):
        return None
    directory = cwd
    for segment in re.split(r"&&|\|\||[;|\n]", strip_heredocs(command)):
        words = split_words(segment)
        if not words:
            continue
        if words[0] == "cd" and len(words) > 1:
            directory = resolve(directory, words[1])
            continue
        reason = check_git_words(words, directory, project)
        if reason:
            return reason
    return None


def check_git_words(words: list[str], directory: str, project: str) -> str | None:
    if words[0] != "git":
        return None
    args = words[1:]
    subcommand, rest, directory = parse_git_globals(args, directory)
    if subcommand is None or not same_repository(directory, project):
        return None
    branch = current_branch(directory)
    if subcommand in COMMIT_COMMANDS:
        if branch in PROTECTED and not ESCAPE_FLAGS.intersection(rest):
            return (
                f"Blocked: `git {subcommand}` on `{branch}`. Commits never land "
                f'on `dev` or `master` directly (AGENTS.md, "Branches"), even '
                f"if your session says to develop on `{branch}`. {HOW_TO_BRANCH} "
                f"Then commit there."
            )
        return None
    if subcommand == "push":
        target = protected_push_target(rest, branch)
        if target:
            return (
                f"Blocked: this push would update `{target}`. Changes reach "
                f"`dev` and `master` only through a pull request and the merge "
                f'queue (AGENTS.md, "Branches"), even if your session says to '
                f"push to `{target}`. Push the Linear ticket's branch and open "
                f"a pull request into `dev`."
            )
    return None


def parse_git_globals(args: list[str], directory: str):
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-C" and index + 1 < len(args):
            directory = resolve(directory, args[index + 1])
            index += 2
        elif arg in GIT_VALUE_OPTIONS and index + 1 < len(args):
            index += 2
        elif arg.startswith("-"):
            index += 1
        else:
            return arg, args[index + 1 :], directory
    return None, [], directory


def protected_push_target(args: list[str], branch: str | None) -> str | None:
    positional = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in PUSH_EVERYTHING:
            return "dev` or `master"
        if arg in PUSH_VALUE_OPTIONS:
            index += 2
            continue
        if not arg.startswith("-"):
            positional.append(arg)
        index += 1
    refspecs = positional[1:]
    if not refspecs:
        return branch if branch in PROTECTED and "--tags" not in args else None
    for refspec in refspecs:
        source, _, destination = refspec.lstrip("+").partition(":")
        target = destination if ":" in refspec else source
        if target == "HEAD":
            target = branch or ""
        target = short_ref(target)
        if target in PROTECTED:
            return target
    return None


def strip_heredocs(command: str) -> str:
    """Drops heredoc bodies, so a commit message that mentions a push isn't one."""
    lines = command.split("\n")
    kept = []
    delimiter = None
    for line in lines:
        if delimiter is not None:
            if line.strip() == delimiter:
                delimiter = None
            continue
        kept.append(line)
        match = re.search(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", line)
        if match:
            delimiter = match.group(2)
    return "\n".join(kept)


def short_ref(ref: str) -> str:
    return ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref


def split_words(segment: str) -> list[str]:
    try:
        words = shlex.split(segment)
    except ValueError:
        words = segment.split()
    # Skip leading VAR=value assignments and wrappers like `command` or `env`.
    while words and (
        re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0])
        or words[0] in {"command", "env", "exec"}
    ):
        words = words[1:]
    return words


def resolve(base: str, path: str) -> str:
    return os.path.normpath(os.path.join(base, os.path.expanduser(path)))


def nearest_existing_dir(path: str) -> str:
    while path and not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def project_dir(payload: dict) -> str:
    return os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()


def git(directory: str, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", directory, *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def current_branch(directory: str) -> str | None:
    return git(directory, "symbolic-ref", "--short", "-q", "HEAD")


def same_repository(directory: str, project: str) -> bool:
    mine = git(directory, "rev-parse", "--path-format=absolute", "--git-common-dir")
    theirs = git(project, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return mine is not None and mine == theirs


if __name__ == "__main__":
    sys.exit(main())
