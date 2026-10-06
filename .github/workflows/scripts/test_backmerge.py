"""backmerge.sh against throwaway repositories: a bare `origin`, a clone that
writes the history, and a clone the script runs in, as the workflow's checkout.
`gh` is a fake on PATH that records each call and answers from `self.open_prs`.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

SCRIPT = Path(__file__).with_name("backmerge.sh")
REAL_GIT = shutil.which("git") or "git"
REPO = "example/repo"

FAKE_GH = """#!/usr/bin/env python3
import json, os, sys
state = os.environ["FAKE_GH_STATE"]
body = sys.stdin.read() if "--body-file" in sys.argv else None
with open(os.path.join(state, "calls.jsonl"), "a") as log:
    log.write(json.dumps({"args": sys.argv[1:], "body": body}) + "\\n")
if sys.argv[1] == "api":
    with open(os.path.join(state, "api.json")) as f:
        answers = json.load(f)
    path = sys.argv[-1]
    print(json.dumps(answers["comments"] if "/comments" in path else answers["pulls"]))
"""

# Advances origin's dev the first time the script pushes to dev, as a merge
# landing on dev between the script's fetch and its push would.
RACING_GIT = """#!/usr/bin/env bash
if [[ " $* " == *":refs/heads/dev "* && ! -e "$RACE_MARKER" ]]; then
  touch "$RACE_MARKER"
  "$REAL_GIT" -C "$RACE_WORK" push -q origin dev
fi
exec "$REAL_GIT" "$@"
"""

REFUSE_DEV = """#!/bin/sh
while read old new ref; do
  if [ "$ref" = refs/heads/dev ]; then
    echo "GH013: Repository rule violations found for refs/heads/dev." >&2
    exit 1
  fi
done
"""


class BackmergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.make_repos()

    def make_repos(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="backmerge-test-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.state = self.root / "gh"
        self.state.mkdir()
        (self.bin / "gh").write_text(FAKE_GH)
        (self.bin / "gh").chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "backmerge-bot[bot]",
            "GIT_AUTHOR_EMAIL": "backmerge-bot[bot]@users.noreply.github.com",
            "GIT_COMMITTER_NAME": "backmerge-bot[bot]",
            "GIT_COMMITTER_EMAIL": "backmerge-bot[bot]@users.noreply.github.com",
            "GITHUB_REPOSITORY": REPO,
            "FAKE_GH_STATE": str(self.state),
        }
        self.open_prs: List[Dict] = []
        self.comments: List[Dict] = []

        self.origin = self.root / "origin.git"
        self.git(self.root, "init", "-q", "--bare", "-b", "master", str(self.origin))
        self.work = self.root / "work"
        self.git(self.root, "clone", "-q", str(self.origin), str(self.work))
        (self.work / "app.txt").write_text("one\ntwo\nthree\n")
        (self.work / "notes.txt").write_text("a\n")
        self.git(self.work, "add", "-A")
        self.git(self.work, "commit", "-q", "-m", "base")
        self.git(self.work, "push", "-q", "origin", "master", "master:dev")

    def git(self, cwd: Path, *args: str) -> str:
        return subprocess.run(
            [REAL_GIT, *args],
            cwd=cwd,
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(
        self, branch: str, files: Dict[str, str], message: str, push: bool = True
    ) -> str:
        self.git(self.work, "fetch", "-q", "origin")
        if self.git(self.work, "branch", "--list", branch):
            self.git(self.work, "switch", "-q", branch)
        else:
            self.git(self.work, "switch", "-q", "-c", branch, f"origin/{branch}")
        for name, content in files.items():
            (self.work / name).write_text(content)
        self.git(self.work, "add", "-A")
        self.git(self.work, "commit", "-q", "-m", message)
        if push:
            self.git(self.work, "push", "-q", "origin", branch)
        return self.git(self.work, "rev-parse", "HEAD")

    def ref(self, name: str) -> str:
        """The sha `name` points at on origin, or "" if there is no such ref."""
        return subprocess.run(
            [REAL_GIT, "rev-parse", "--verify", "--quiet", name],
            cwd=self.origin,
            env=self.env,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def run_script(
        self, *args: str, dry_run: bool = False
    ) -> subprocess.CompletedProcess:
        (self.state / "api.json").write_text(
            json.dumps({"pulls": [self.open_prs], "comments": [self.comments]})
        )
        runner = self.root / "runner"
        if not runner.exists():
            self.git(self.root, "clone", "-q", str(self.origin), str(runner))
        return subprocess.run(
            [str(SCRIPT), *args],
            cwd=runner,
            env={**self.env, "DRY_RUN": "true" if dry_run else "false"},
            capture_output=True,
            text=True,
        )

    def gh_calls(self, *command: str) -> List[Dict]:
        log = self.state / "calls.jsonl"
        calls = (
            [json.loads(l) for l in log.read_text().splitlines()]
            if log.exists()
            else []
        )
        return [c for c in calls if c["args"][: len(command)] == list(command)]

    def parents(self, sha: str) -> List[str]:
        return self.git(self.origin, "rev-list", "--parents", "-n1", sha).split()[1:]

    def conflict(self) -> str:
        """master and dev change the same line; returns master's tip."""
        self.commit("dev", {"app.txt": "one\nDEV\nthree\n"}, "dev edits two")
        return self.commit(
            "master", {"app.txt": "one\nHOTFIX\nthree\n"}, "hotfix two (#1)"
        )

    def test_nothing_to_do_when_dev_already_contains_master(self) -> None:
        dev = self.commit("dev", {"notes.txt": "a\nb\n"}, "dev only")

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already contains", result.stdout)
        self.assertEqual(self.ref("dev"), dev)
        self.assertEqual(self.gh_calls(), [])

    def test_merge_is_pushed_to_dev_with_dev_as_first_parent(self) -> None:
        cases = {
            "fast-forward": {},
            "clean merge": {"notes.txt": "a\nfrom dev\n"},
        }
        for case, dev_change in cases.items():
            with self.subTest(case):
                self.make_repos()
                dev = (
                    self.commit("dev", dev_change, "dev change")
                    if dev_change
                    else self.ref("dev")
                )
                master = self.commit(
                    "master", {"app.txt": "one\nhotfix\nthree\n"}, "hotfix (#1)"
                )

                result = self.run_script()

                self.assertEqual(result.returncode, 0, result.stderr)
                merged = self.ref("dev")
                self.assertEqual(self.parents(merged), [dev, master])
                self.assertEqual(
                    self.git(self.origin, "show", f"{merged}:app.txt"),
                    "one\nhotfix\nthree",
                )
                if dev_change:
                    self.assertEqual(
                        self.git(self.origin, "show", f"{merged}:notes.txt"),
                        "a\nfrom dev",
                    )
                message = self.git(self.origin, "log", "-1", "--format=%B", merged)
                self.assertIn(master, message)
                self.assertIn("hotfix (#1)", message)
                self.assertEqual(self.gh_calls("pr"), [])

    def test_conflict_opens_one_draft_pr_and_leaves_dev_alone(self) -> None:
        master = self.conflict()
        dev = self.ref("dev")
        branch = f"backmerge/master-{master[:12]}"

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.ref("dev"), dev)
        self.assertEqual(self.ref(branch), master)
        [create] = self.gh_calls("pr", "create")
        args = create["args"]
        for flag, value in [
            ("--base", "dev"),
            ("--head", branch),
            ("--label", "conflicts-help"),
        ]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--draft", args)
        self.assertIn("- `app.txt`", create["body"])
        self.assertIn("CONFLICT (content): Merge conflict in app.txt", create["body"])
        self.assertIn("hotfix two (#1)", create["body"])
        self.assertIn("dev edits two", create["body"])

        self.open_prs = [
            {"number": 7, "head": {"ref": branch, "repo": {"full_name": REPO}}}
        ]
        again = self.run_script()

        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("#7 already carries", again.stdout)
        self.assertEqual(len(self.gh_calls("pr", "create")), 1)
        self.assertEqual(self.gh_calls("pr", "comment"), [])

    def test_open_pr_is_told_once_when_master_moves(self) -> None:
        first = self.conflict()
        self.git(
            self.work,
            "push",
            "-q",
            "origin",
            f"{first}:refs/heads/backmerge/master-{first[:12]}",
        )
        self.open_prs = [
            {
                "number": 8,
                "head": {
                    "ref": "backmerge/master-elsewhere",
                    "repo": {"full_name": "fork/repo"},
                },
            },
            {
                "number": 7,
                "head": {
                    "ref": f"backmerge/master-{first[:12]}",
                    "repo": {"full_name": REPO},
                },
            },
        ]
        second = self.commit(
            "master", {"notes.txt": "a\nsecond hotfix\n"}, "hotfix (#2)"
        )

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        [comment] = self.gh_calls("pr", "comment")
        self.assertEqual(comment["args"][2], "7")
        self.assertIn(f"<!-- backmerge:{second} -->", comment["body"])
        self.assertEqual(self.gh_calls("pr", "create"), [])

        self.comments = [{"body": comment["body"]}]
        again = self.run_script()

        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(len(self.gh_calls("pr", "comment")), 1)

    def test_dry_run_pushes_nothing_and_opens_nothing(self) -> None:
        self.commit("master", {"notes.txt": "a\nhotfix\n"}, "clean hotfix (#1)")
        dev = self.ref("dev")

        clean = self.run_script(dry_run=True)

        self.assertEqual(clean.returncode, 0, clean.stderr)
        self.assertIn("Would push this merge", clean.stdout)
        self.assertEqual(self.ref("dev"), dev)

        master = self.conflict()
        dev = self.ref("dev")

        conflicted = self.run_script(dry_run=True)

        self.assertEqual(conflicted.returncode, 0, conflicted.stderr)
        self.assertIn("open a draft PR", conflicted.stdout)
        self.assertEqual(self.ref("dev"), dev)
        self.assertEqual(self.ref(f"backmerge/master-{master[:12]}"), "")
        self.assertEqual(self.gh_calls("pr"), [])

    def test_a_refused_push_says_what_the_admin_must_configure(self) -> None:
        self.commit("master", {"notes.txt": "a\nhotfix\n"}, "hotfix (#1)")
        dev = self.ref("dev")
        hook = self.origin / "hooks" / "pre-receive"
        hook.write_text(REFUSE_DEV)
        hook.chmod(0o755)

        result = self.run_script()

        self.assertEqual(result.returncode, 1)
        self.assertIn("bypass list", result.stderr)
        self.assertEqual(self.ref("dev"), dev)

    def test_merges_again_when_dev_moves_during_the_push(self) -> None:
        master = self.commit("master", {"notes.txt": "a\nhotfix\n"}, "hotfix (#1)")
        late = self.commit(
            "dev",
            {"other.txt": "landed during the push\n"},
            "late dev merge",
            push=False,
        )
        (self.bin / "git").write_text(RACING_GIT)
        (self.bin / "git").chmod(0o755)
        self.env.update(
            REAL_GIT=REAL_GIT,
            RACE_WORK=str(self.work),
            RACE_MARKER=str(self.root / "raced"),
        )

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dev moved during the push", result.stdout)
        self.assertEqual(self.parents(self.ref("dev")), [late, master])

    def test_verify_accepts_a_union_and_names_an_invented_line(self) -> None:
        master = self.conflict()
        runner = self.root / "runner"
        self.run_script(dry_run=True)
        self.git(runner, "switch", "-q", "-c", "resolve", "origin/dev")

        resolutions = {
            "union": ("one\nDEV\nHOTFIX\nthree\n", 0, None),
            "invented": ("one\nDEV and HOTFIX\nthree\n", 1, "| DEV and HOTFIX"),
        }
        for case, (content, code, named) in resolutions.items():
            with self.subTest(case):
                self.git(runner, "reset", "-q", "--hard", "resolve")
                subprocess.run(
                    [REAL_GIT, "merge", "-q", master],
                    cwd=runner,
                    env=self.env,
                    capture_output=True,
                )
                (runner / "app.txt").write_text(content)
                self.git(runner, "commit", "-q", "-am", "resolve")

                result = self.run_script("verify", "HEAD", "origin/dev", master)

                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                if named:
                    self.assertIn(named, result.stdout)
                self.assertIn("blend: app.txt", result.stdout)


if __name__ == "__main__":
    unittest.main()
