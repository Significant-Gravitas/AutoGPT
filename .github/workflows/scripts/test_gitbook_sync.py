import contextlib
import io
import json
import os
import re
import unittest
import urllib.error
import urllib.parse
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import gitbook_sync
from gitbook_sync import ISSUE_TITLE, ApiError, GitHub, snapshot_message, synced_from

BOT = "docs-sync[bot]"
PERSON = {
    "name": "Some Editor",
    "email": "e@example.com",
    "date": "2026-10-03T00:00:00Z",
}
GITBOOK_BOT = {
    "name": "gitbook-bot",
    "email": "ghost@gitbook.com",
    "date": "2026-10-03T00:00:00Z",
}


class FakeGitHub:
    """The slice of the GitHub API the sync uses, over in-memory objects."""

    repo = "example/repo"
    actor = BOT

    def __init__(self) -> None:
        self.commits: Dict[str, Dict[str, Any]] = {}
        self.trees: Dict[str, List[Dict[str, str]]] = {}
        self.refs: Dict[str, str] = {}
        self.issues: List[Dict[str, Any]] = []
        self.comments: List[Dict[str, Any]] = []
        self.activity: List[Dict[str, Any]] = []
        self.writes: List[str] = []
        self.reject_ref_update = False
        self.reject_issue_writes = False
        self._next = 0

    def _sha(self) -> str:
        self._next += 1
        return f"{self._next:040x}"

    def tree(self, **entries: str) -> str:
        """Add a root tree; `docs="x"` makes a docs/ directory with content x."""
        sha = self._sha()
        self.trees[sha] = [
            {"path": path, "type": "tree", "sha": f"{content:0>40}"}
            for path, content in entries.items()
        ]
        return sha

    def commit(
        self,
        tree: str,
        parents: List[str],
        message: str = "a change",
        committer: Optional[Dict[str, str]] = None,
    ) -> str:
        sha = self._sha()
        self.commits[sha] = {
            "sha": sha,
            "message": message,
            "tree": {"sha": tree},
            "parents": [{"sha": p} for p in parents],
            "author": PERSON,
            "committer": committer or PERSON,
        }
        return sha

    def move(
        self, branch: str, sha: str, login: Optional[str], kind: str = "push"
    ) -> None:
        """Point a branch at a commit the way GitHub records it."""
        self.refs[branch] = sha
        self.activity.insert(
            0,
            {
                "timestamp": "2026-10-03T00:00:00Z",
                "activity_type": kind,
                "actor": {"login": login} if login else None,
                "ref": f"refs/heads/{branch}",
                "after": sha,
            },
        )

    def open_issue(self, title: str, body: str, login: str) -> Dict[str, Any]:
        issue = {
            "number": len(self.issues) + 1,
            "title": title,
            "body": body,
            "state": "open",
            "user": {"login": login},
            "html_url": f"https://example.com/issues/{len(self.issues) + 1}",
        }
        self.issues.append(issue)
        return issue

    def call(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        retry: bool = True,
    ) -> Any:
        if method != "GET":
            self.writes.append(f"{method} {path}")
        path, _, query = path.partition("?")
        params = dict(urllib.parse.parse_qsl(query))
        if m := re.fullmatch(r"/git/ref/heads/(.+)", path):
            if m[1] not in self.refs:
                raise ApiError(method, path, 404, "Not Found")
            return {"object": {"sha": self.refs[m[1]]}}
        if m := re.fullmatch(r"/git/commits/(.+)", path):
            if m[1] not in self.commits:
                raise ApiError(method, path, 404, "Not Found")
            return self.commits[m[1]]
        if m := re.fullmatch(r"/git/trees/(.+)", path):
            return {"tree": self.trees[m[1]]}
        if (method, path) == ("POST", "/git/trees"):
            sha = self._sha()
            self.trees[sha] = [
                {"path": e["path"], "type": e["type"], "sha": e["sha"]}
                for e in body["tree"]
            ]
            return {"sha": sha}
        if (method, path) == ("POST", "/git/commits"):
            return {"sha": self.commit(body["tree"], body["parents"], body["message"])}
        if m := re.fullmatch(r"/git/refs/heads/(.+)", path):
            if self.reject_ref_update or body["force"]:
                raise ApiError(method, path, 422, "Update is not a fast forward")
            self.move(m[1], body["sha"], self.actor)
            return {}
        if path == "/activity":
            return [
                a
                for a in self.activity
                if a["ref"] == params["ref"]
                and (
                    "actor" not in params
                    or (a["actor"] or {}).get("login") == params["actor"]
                )
            ]
        if (method, path) == ("GET", "/issues"):
            return [
                i
                for i in self.issues
                if i["state"] == "open" and i["user"]["login"] == params["creator"]
            ]
        if self.reject_issue_writes and path.startswith("/issues"):
            raise ApiError(method, path, 503, "Service Unavailable")
        if (method, path) == ("POST", "/issues"):
            return self.open_issue(body["title"], body["body"], self.actor)
        if m := re.fullmatch(r"/issues/(\d+)/comments", path):
            if method == "GET":
                return [c for c in self.comments if c["issue"] == int(m[1])]
            self.comments.append({"issue": int(m[1]), "body": body["body"]})
            return {}
        raise AssertionError(f"unexpected call: {method} {path}")


def run(
    gh: FakeGitHub, dry_run: bool = False, output: Optional[io.StringIO] = None
) -> int:
    # Keep the script's workflow commands and job summary out of the CI job
    # that runs these tests.
    quiet = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")}
    with patch.dict(os.environ, quiet, clear=True):
        with contextlib.redirect_stdout(output or io.StringIO()):
            return gitbook_sync.run(
                gh, "master", "gitbook", notify="@example/team", dry_run=dry_run
            )


class SyncedFromTests(unittest.TestCase):
    def test_reads_trailer(self):
        sha = "a" * 40
        self.assertEqual(synced_from(snapshot_message("master", sha)), sha)

    def test_ignores_mentions_that_are_not_a_trailer_line(self):
        self.assertIsNone(synced_from(f"Revert the commit Synced-From: {'a' * 40}"))
        self.assertIsNone(synced_from("Synced-From: not-a-sha"))


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.gh = FakeGitHub()
        # master carries code and docs; gitbook starts as an old full copy
        # with a few commits of its own.
        self.master = self.gh.commit(self.gh.tree(docs="1", autogpt_platform="c"), [])
        self.gh.move("master", self.master, "releaser")
        older = self.gh.commit(self.gh.tree(docs="0", autogpt_platform="a"), [])
        self.legacy = self.gh.commit(
            self.gh.tree(docs="0", autogpt_platform="b"), [older]
        )
        self.gh.move("gitbook", self.legacy, "someone")

    def head(self) -> Dict[str, Any]:
        return self.gh.commits[self.gh.refs["gitbook"]]

    def advance_master(self, docs: str) -> str:
        self.master = self.gh.commit(
            self.gh.tree(docs=docs, autogpt_platform="c"), [self.master]
        )
        self.gh.move("master", self.master, "releaser")
        return self.master

    def stray(self, message: str = "a stray edit", docs: str = "edited") -> str:
        """Somebody else commits on top of gitbook."""
        sha = self.gh.commit(
            self.gh.tree(docs=docs), [self.gh.refs["gitbook"]], message
        )
        self.gh.move("gitbook", sha, "someone")
        return sha

    def test_first_sync_writes_docs_only_snapshot_without_reporting(self):
        self.assertEqual(run(self.gh), 0)

        head = self.head()
        self.assertEqual([p["sha"] for p in head["parents"]], [self.legacy])
        self.assertEqual(synced_from(head["message"]), self.master)
        self.assertEqual(
            self.gh.trees[head["tree"]["sha"]],
            [{"path": "docs", "type": "tree", "sha": f"{'1':0>40}"}],
        )
        self.assertEqual(self.gh.issues, [])

    def test_nothing_to_do_when_snapshot_is_current(self):
        run(self.gh)
        self.gh.writes.clear()

        self.assertEqual(run(self.gh), 0)

        self.assertEqual(self.gh.writes, [])

    def test_master_moving_without_docs_changes_writes_nothing(self):
        run(self.gh)
        self.advance_master(docs="1")
        self.gh.writes.clear()

        self.assertEqual(run(self.gh), 0)

        self.assertEqual(self.gh.writes, [])

    def test_docs_change_on_master_fast_forwards_gitbook(self):
        run(self.gh)
        previous = self.gh.refs["gitbook"]
        self.advance_master(docs="2")

        self.assertEqual(run(self.gh), 0)

        head = self.head()
        self.assertEqual([p["sha"] for p in head["parents"]], [previous])
        self.assertEqual(synced_from(head["message"]), self.master)
        self.assertEqual(self.gh.issues, [])

    def test_foreign_commit_is_reported_and_docs_are_restored(self):
        run(self.gh)
        snapshot = self.gh.refs["gitbook"]
        foreign = self.gh.commit(
            self.gh.tree(docs="edited"),
            [snapshot],
            "GITBOOK-101: tweak install steps",
            committer=GITBOOK_BOT,
        )
        self.gh.move("gitbook", foreign, "gitbook-com[bot]")

        self.assertEqual(run(self.gh), 1)

        head = self.head()
        self.assertEqual([p["sha"] for p in head["parents"]], [foreign])
        self.assertEqual(self.gh.trees[head["tree"]["sha"]][0]["sha"], f"{'1':0>40}")
        self.assertEqual(len(self.gh.issues), 1)
        issue = self.gh.issues[0]
        self.assertEqual(issue["title"], ISSUE_TITLE)
        self.assertIn(foreign, issue["body"])
        self.assertIn("GITBOOK-101: tweak install steps", issue["body"])
        self.assertIn("committed by gitbook-bot", issue["body"])
        self.assertIn("push by `gitbook-com[bot]`", issue["body"])
        self.assertIn("cc @example/team", issue["body"])
        self.assertNotIn(snapshot, issue["body"])
        self.assertNotIn(BOT, issue["body"])

    def test_report_is_filed_before_the_branch_moves(self):
        """Otherwise a failed report could never be retried."""
        run(self.gh)
        stray = self.stray()
        self.gh.reject_issue_writes = True

        self.assertEqual(run(self.gh), 1)
        self.assertEqual(self.gh.refs["gitbook"], stray)

        self.gh.reject_issue_writes = False
        self.assertEqual(run(self.gh), 1)
        self.assertNotEqual(self.gh.refs["gitbook"], stray)
        self.assertIn("a stray edit", self.gh.issues[0]["body"])

    def test_second_report_comments_on_the_open_issue(self):
        run(self.gh)
        for subject in ("first stray", "second stray"):
            self.stray(subject)
            run(self.gh)

        self.assertEqual(len(self.gh.issues), 1)
        self.assertEqual(len(self.gh.comments), 1)
        self.assertIn("second stray", self.gh.comments[0]["body"])
        self.assertNotIn("first stray", self.gh.comments[0]["body"])

    def test_issue_opened_by_someone_else_is_not_reused(self):
        run(self.gh)
        self.gh.open_issue(ISSUE_TITLE, "not from the sync", "an-outsider")
        self.stray()

        run(self.gh)

        self.assertEqual(self.gh.comments, [])
        self.assertEqual(
            [i["user"]["login"] for i in self.gh.issues], ["an-outsider", BOT]
        )

    def test_merge_is_reported_as_one_commit(self):
        run(self.gh)
        snapshot = self.gh.refs["gitbook"]
        merged_in = self.gh.commit(self.gh.tree(docs="dev"), [], "work from dev")
        merge = self.gh.commit(
            self.gh.tree(docs="dev", autogpt_platform="d"),
            [snapshot, merged_in],
            "Merge branch 'dev' into gitbook",
        )
        self.gh.move("gitbook", merge, "someone", "pr_merge")

        run(self.gh)

        body = self.gh.issues[0]["body"]
        self.assertIn(merge, body)
        self.assertNotIn(merged_in, body)
        self.assertIn("pr_merge by `someone`", body)

    def test_snapshot_with_changed_content_is_foreign(self):
        """An amended snapshot keeps the trailer but not the content."""
        run(self.gh)
        amended = self.stray(snapshot_message("master", self.master))

        run(self.gh)

        self.assertIn(amended, self.gh.issues[0]["body"])

    def test_snapshot_naming_a_missing_commit_is_foreign(self):
        run(self.gh)
        forged = self.stray(snapshot_message("master", "f" * 40), docs="1")

        run(self.gh)

        self.assertIn(forged, self.gh.issues[0]["body"])

    def test_trailer_below_the_head_does_not_hide_a_foreign_commit(self):
        run(self.gh)
        amended = self.stray(snapshot_message("master", self.master))
        on_top = self.stray("on top of it")

        run(self.gh)

        body = self.gh.issues[0]["body"]
        self.assertIn(on_top, body)
        self.assertIn(amended, body)

    def test_branch_replaced_after_a_sync_is_reported(self):
        """Recreated or force-pushed: no snapshot left to compare against."""
        run(self.gh)
        self.gh.move("gitbook", self.master, "an-admin", "force_push")

        self.assertEqual(run(self.gh), 1)

        body = self.gh.issues[0]["body"]
        self.assertIn(self.master, body)
        self.assertIn("rewritten, recreated", body)
        self.assertIn("force_push by `an-admin`", body)
        self.assertEqual(synced_from(self.head()["message"]), self.master)

    def test_more_foreign_commits_than_the_history_limit_are_reported(self):
        run(self.gh)
        for n in range(5):
            self.stray(f"stray {n}")

        with patch.object(gitbook_sync, "HISTORY_LIMIT", 3):
            self.assertEqual(run(self.gh), 1)

        body = self.gh.issues[0]["body"]
        self.assertIn("stray 4", body)
        self.assertIn("rewritten, recreated", body)

    def test_commit_text_cannot_mention_or_format(self):
        run(self.gh)
        self.stray("fix `x` @someone [link](https://example.com)")

        run(self.gh)

        self.assertIn(
            "`fix 'x' @someone [link](https://example.com)`", self.gh.issues[0]["body"]
        )

    def test_control_characters_in_a_name_cannot_start_a_new_line(self):
        """A carriage return would begin a workflow command in the job log."""
        run(self.gh)
        stray = self.gh.commit(
            self.gh.tree(docs="edited"), [self.gh.refs["gitbook"]], "a stray edit"
        )
        self.gh.commits[stray]["author"] = dict(
            PERSON, name="Eve\r::error::forged\r\r@victim [l](https://example.com)"
        )
        self.gh.move("gitbook", stray, "someone")

        output = io.StringIO()
        run(self.gh, output=output)

        self.assertNotIn("\r", output.getvalue())
        self.assertNotIn("\r", self.gh.issues[0]["body"])
        self.assertIn(
            "`Eve ::error::forged @victim [l](https://example.com), "
            "committed by Some Editor`",
            self.gh.issues[0]["body"],
        )

    def test_bot_activity_on_another_branch_is_not_a_previous_sync(self):
        self.gh.move("batch/rollup", self.master, BOT, "force_push")

        self.assertEqual(run(self.gh), 0)

        self.assertEqual(self.gh.issues, [])

    def test_pull_request_with_the_report_title_is_not_reused(self):
        """The issues endpoint lists pull requests too."""
        run(self.gh)
        self.gh.open_issue(ISSUE_TITLE, "a pull request", BOT)["pull_request"] = {}
        self.stray()

        run(self.gh)

        self.assertEqual(self.gh.comments, [])
        self.assertEqual(len(self.gh.issues), 2)

    def test_persistent_failure_is_reported_once(self):
        self.gh.reject_ref_update = True

        for _ in range(5):
            self.assertEqual(run(self.gh), 1)

        self.assertEqual(len(self.gh.issues), 1)
        self.assertEqual(self.gh.comments, [])

    def test_foreign_commits_with_a_failing_write_are_reported_once(self):
        run(self.gh)
        self.stray()
        self.gh.reject_ref_update = True

        for _ in range(5):
            self.assertEqual(run(self.gh), 1)

        self.assertEqual(len(self.gh.issues), 1)
        self.assertIn("a stray edit", self.gh.issues[0]["body"])
        self.assertEqual(len(self.gh.comments), 1)
        self.assertIn("could not update `gitbook`", self.gh.comments[0]["body"])

    def test_issue_posts_are_not_retried(self):
        """A retried POST that had succeeded would open a second issue."""
        run(self.gh)
        self.stray()
        calls = []
        real_call = self.gh.call

        def recording(method, path, body=None, retry=True):
            calls.append((method, path.partition("?")[0], retry))
            return real_call(method, path, body, retry)

        self.gh.call = recording
        run(self.gh)

        self.assertIn(("POST", "/issues", False), calls)
        self.assertIn(("PATCH", "/git/refs/heads/gitbook", True), calls)

    def test_activity_without_an_actor_does_not_stop_the_restore(self):
        run(self.gh)
        stray = self.stray()
        self.gh.activity[0]["actor"] = None

        self.assertEqual(run(self.gh), 1)

        self.assertNotEqual(self.gh.refs["gitbook"], stray)
        self.assertIn("push by `unknown`", self.gh.issues[0]["body"])

    def test_dry_run_writes_nothing(self):
        self.assertEqual(run(self.gh, dry_run=True), 0)

        self.assertEqual(self.gh.writes, [])
        self.assertEqual(self.gh.refs["gitbook"], self.legacy)

    def test_source_without_docs_is_reported_and_fails(self):
        self.gh.move(
            "master",
            self.gh.commit(self.gh.tree(autogpt_platform="c"), [self.master]),
            "releaser",
        )

        self.assertEqual(run(self.gh), 1)

        self.assertEqual(self.gh.refs["gitbook"], self.legacy)
        self.assertIn("has no docs/ directory", self.gh.issues[0]["body"])

    def test_missing_target_branch_is_reported_and_fails(self):
        del self.gh.refs["gitbook"]

        self.assertEqual(run(self.gh), 1)

        self.assertIn("could not update `gitbook`", self.gh.issues[0]["body"])
        self.assertIn("HTTP 404", self.gh.issues[0]["body"])

    def test_rejected_ref_update_is_reported_and_fails(self):
        """A sync that cannot write must not just leave the site stale."""
        self.gh.reject_ref_update = True

        self.assertEqual(run(self.gh), 1)

        self.assertEqual(self.gh.refs["gitbook"], self.legacy)
        self.assertEqual(len(self.gh.issues), 1)
        self.assertIn("could not update `gitbook`", self.gh.issues[0]["body"])
        self.assertIn("Update is not a fast forward", self.gh.issues[0]["body"])


def response(payload: Any) -> MagicMock:
    opened = MagicMock()
    opened.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
    return opened


def http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "url", status, "error", {}, io.BytesIO(b"upstream\nbroke")
    )


class GitHubCallTests(unittest.TestCase):
    def setUp(self):
        self.gh = GitHub("https://api.example.com", "example/repo", "token", BOT)
        self.gh.retry_delay = 0

    def test_retries_a_server_error(self):
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = [http_error(502), response({"ok": True})]

            self.assertEqual(self.gh.call("GET", "/git/ref/heads/master"), {"ok": True})

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(
            urlopen.call_args.args[0].full_url,
            "https://api.example.com/repos/example/repo/git/ref/heads/master",
        )

    def test_does_not_retry_a_rejection(self):
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = [http_error(422), response({})]

            with self.assertRaises(ApiError) as raised:
                self.gh.call("PATCH", "/git/refs/heads/gitbook", {"sha": "x"})

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(raised.exception.status, 422)
        self.assertIn("upstream broke", str(raised.exception))

    def test_reply_that_is_not_json_becomes_an_api_error(self):
        broken = MagicMock()
        broken.__enter__.return_value = io.BytesIO(b"")
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = broken

            with self.assertRaises(ApiError) as raised:
                self.gh.call("GET", "/git/ref/heads/master")

        self.assertEqual(raised.exception.status, 0)

    def test_unrepeatable_call_is_sent_once(self):
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = [http_error(502), response({})]

            with self.assertRaises(ApiError):
                self.gh.call("POST", "/issues", {"title": "t"}, retry=False)

        self.assertEqual(urlopen.call_count, 1)

    def test_network_failure_becomes_an_api_error(self):
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.side_effect = urllib.error.URLError("connection reset")

            with self.assertRaises(ApiError):
                self.gh.call("GET", "/git/ref/heads/master")

        self.assertEqual(urlopen.call_count, gitbook_sync.ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
