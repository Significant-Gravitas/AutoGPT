"""Keep the `gitbook` branch a one-way, docs-only copy of master's docs/.

GitBook publishes agpt.co/docs from the `gitbook` branch. Nothing else should
write to that branch: a ruleset restricts updates to the App this runs as. Each
run that finds a difference writes one commit on top of `gitbook` whose tree is
exactly master's docs/ directory, so the branch only ever fast-forwards.

A commit this script wrote names the master commit it copied in a
`Synced-From:` trailer. If the branch head is anything else, something other
than this script wrote to the branch. The run reports those commits in an
issue, then writes a fresh snapshot on top, which puts master's docs back.
The foreign commits stay in the branch history; nothing is force-pushed.
"""

import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

DOCS_PATH = "docs"
SYNC_TRAILER = "Synced-From"
TRAILER_PATTERN = re.compile(rf"^{SYNC_TRAILER}: ([0-9a-f]{{40}})$", re.MULTILINE)
# How far back to look for the last snapshot when the head is not one.
HISTORY_LIMIT = 100
MAX_REPORTED = 20
ISSUE_TITLE = "Docs sync: the `gitbook` branch needs attention"
ATTEMPTS = 3
RETRIED_STATUSES = {429, 500, 502, 503, 504}
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class SyncError(Exception):
    """The sync cannot go on. Reported in the issue, then the run fails."""


class ApiError(SyncError):
    def __init__(self, method: str, path: str, status: int, body: str) -> None:
        super().__init__(
            f"{method} {path} -> HTTP {status}: {' '.join(body.split())[:500]}"
        )
        self.status = status


class GitHub:
    retry_delay = 2.0

    def __init__(self, api_url: str, repo: str, token: str, actor: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.repo = repo
        self.token = token
        # The login the token acts as, e.g. "autogpt-batch-bot[bot]".
        self.actor = actor

    def call(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        retry: bool = True,
    ) -> Any:
        """Call the API. `retry=False` is for calls that are not safe to
        repeat: a retried POST that had in fact succeeded posts twice."""
        request = urllib.request.Request(
            f"{self.api_url}/repos/{self.repo}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "autogpt-gitbook-sync",
            },
        )
        attempts = ATTEMPTS if retry else 1
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                error = ApiError(
                    method, path, e.code, e.read().decode(errors="replace")
                )
                if e.code not in RETRIED_STATUSES:
                    raise error from e
            except (OSError, http.client.HTTPException, ValueError) as e:
                # Connection errors, timeouts, and replies cut short or not JSON.
                error = ApiError(method, path, 0, f"{type(e).__name__}: {e}")
            if attempt < attempts:
                time.sleep(self.retry_delay * attempt)
        raise error


class Snapshot(NamedTuple):
    """What a commit holds, as far as the sync cares."""

    sha: str
    docs_tree: Optional[str]
    docs_only: bool
    synced_from: Optional[str]


def synced_from(message: str) -> Optional[str]:
    """Return the master SHA a snapshot commit says it copied, if any."""
    matches = TRAILER_PATTERN.findall(message)
    return matches[-1] if matches else None


def branch_head(gh: GitHub, branch: str) -> str:
    return gh.call("GET", f"/git/ref/heads/{branch}")["object"]["sha"]


def read_snapshot(gh: GitHub, sha: str) -> Snapshot:
    commit = gh.call("GET", f"/git/commits/{sha}")
    entries = gh.call("GET", f"/git/trees/{commit['tree']['sha']}")["tree"]
    docs = [e for e in entries if e["path"] == DOCS_PATH and e["type"] == "tree"]
    return Snapshot(
        sha=sha,
        docs_tree=docs[0]["sha"] if docs else None,
        docs_only=len(entries) == 1 and bool(docs),
        synced_from=synced_from(commit["message"]),
    )


def is_own_snapshot(gh: GitHub, target: Snapshot) -> bool:
    """True if `target` is exactly the docs/ of the commit its trailer names.

    The trailer alone is not enough: an amended or cherry-picked snapshot
    keeps the trailer while carrying different content.
    """
    if not (target.synced_from and target.docs_only):
        return False
    try:
        named = read_snapshot(gh, target.synced_from)
    except ApiError as e:
        if e.status in (404, 422):
            return False
        raise
    return named.docs_tree == target.docs_tree


def foreign_commits(gh: GitHub, head: str) -> Tuple[List[Dict[str, Any]], bool]:
    """First-parent commits on top of the last snapshot, newest first.

    `head` is known not to be a snapshot, so it is foreign whatever its
    message says. Following first parents keeps a merge to one entry instead
    of every commit it brought in.

    The second value says whether a snapshot was reached. If not, the list is
    only the newest HISTORY_LIMIT commits and the branch may never have been
    synced, or may have been rewritten since.
    """
    foreign: List[Dict[str, Any]] = []
    sha = head
    for _ in range(HISTORY_LIMIT):
        commit = gh.call("GET", f"/git/commits/{sha}")
        if (
            foreign
            and synced_from(commit["message"])
            and is_own_snapshot(gh, read_snapshot(gh, sha))
        ):
            return foreign, True
        foreign.append(commit)
        if not commit["parents"]:
            break
        sha = commit["parents"][0]["sha"]
    return foreign, False


def has_written_before(gh: GitHub, branch: str) -> bool:
    """Whether the account this runs as has ever updated the branch.

    Tells the first sync, where the branch simply predates the sync, from a
    branch that lost its snapshots by being rewritten or recreated.
    """
    actor = urllib.parse.quote(gh.actor)
    try:
        return bool(
            gh.call(
                "GET", f"/activity?ref=refs/heads/{branch}&actor={actor}&per_page=1"
            )
        )
    except ApiError as e:
        # Unknown is treated as yes: better a needless report than a silent
        # overwrite of somebody's commits.
        print(f"Could not read {branch} activity: {e}")
        return True


def ref_updates(gh: GitHub, branch: str, foreign: List[Dict[str, Any]]) -> List[str]:
    """Who moved the branch since the sync last wrote it, as GitHub recorded.

    Commits name their author; this names the account that pushed or merged
    them, which is what shows how they got past the ruleset. Best effort: the
    report goes out without it if the activity log cannot be read.
    """
    try:
        updates = gh.call("GET", f"/activity?ref=refs/heads/{branch}&per_page=100")
    except ApiError as e:
        print(f"Could not read {branch} activity: {e}")
        return []
    logins = [(u.get("actor") or {}).get("login", "unknown") for u in updates]
    if gh.actor in logins:
        since = range(logins.index(gh.actor))
    else:
        shas = {c["sha"] for c in foreign}
        since = [i for i, u in enumerate(updates) if u.get("after") in shas]
    return [
        f"{updates[i]['timestamp'][:10]} {updates[i]['activity_type']} "
        f"by {inert(logins[i])}"
        for i in list(since)[:MAX_REPORTED]
    ]


def inert(text: str) -> str:
    """Quote text from a commit so it cannot mention, link or format.

    Control characters go too: a carriage return in an author name would end
    the quoted span in the issue and start a new line in the job log, where
    the runner reads a line beginning with "::" as a command.
    """
    flat = " ".join(CONTROL_CHARACTERS.sub(" ", text).split())
    return "`" + flat.replace("`", "'")[:200] + "`"


def describe(commit: Dict[str, Any]) -> str:
    subject = commit["message"].splitlines()[0] if commit["message"] else ""
    author = commit["author"]["name"]
    committer = commit["committer"]["name"]
    who = author if author == committer else f"{author}, committed by {committer}"
    return (
        f"{commit['sha']} {inert(subject)} "
        f"({inert(who)}, {commit['committer']['date'][:10]})"
    )


def snapshot_message(source_branch: str, source_sha: str) -> str:
    return (
        f"docs: sync {DOCS_PATH}/ from {source_branch}@{source_sha[:10]}\n\n"
        f"One-way copy for GitBook, written by docs-gitbook-sync.yml.\n"
        f"Do not commit to this branch: change {DOCS_PATH}/ through a PR to dev.\n\n"
        f"{SYNC_TRAILER}: {source_sha}"
    )


def write_snapshot(
    gh: GitHub, branch: str, parent: str, source_branch: str, source: Snapshot
) -> str:
    tree = gh.call(
        "POST",
        "/git/trees",
        {
            "tree": [
                {
                    "path": DOCS_PATH,
                    "mode": "040000",
                    "type": "tree",
                    "sha": source.docs_tree,
                }
            ]
        },
    )
    # No author or committer: GitHub then signs the commit as the App.
    commit = gh.call(
        "POST",
        "/git/commits",
        {
            "message": snapshot_message(source_branch, source.sha),
            "tree": tree["sha"],
            "parents": [parent],
        },
    )
    # Never forced: if the branch moved since `parent` was read, this fails and
    # the next run starts over from the new head.
    gh.call(
        "PATCH", f"/git/refs/heads/{branch}", {"sha": commit["sha"], "force": False}
    )
    return commit["sha"]


def foreign_section(
    gh: GitHub,
    branch: str,
    source_branch: str,
    foreign: List[Dict[str, Any]],
    reached_snapshot: bool,
) -> List[str]:
    lines = [
        f"`{branch}` is a one-way copy of `{DOCS_PATH}/` on `{source_branch}`, "
        "written only by `.github/workflows/docs-gitbook-sync.yml`. "
        "These commits were written to it by something else:",
        "",
        *(f"- {describe(c)}" for c in foreign[:MAX_REPORTED]),
    ]
    if not reached_snapshot:
        lines.append(
            "- ...and the history below them. None of the sync's own commits "
            f"is within {HISTORY_LIMIT} commits of the head, so the branch was "
            "rewritten, recreated, or had another branch pushed over it."
        )
    elif len(foreign) > MAX_REPORTED:
        lines.append(f"- ...and {len(foreign) - MAX_REPORTED} more")
    updates = ref_updates(gh, branch, foreign)
    if updates:
        lines += ["", "How they got there:", "", *(f"- {u}" for u in updates)]
    return lines + [
        "",
        f"This run now puts `{source_branch}`'s docs back on top of them, which "
        "takes these changes off the published site. The commits stay in the "
        "branch history. If that write fails, a second report follows here.",
        "",
        "To keep a change, open a PR against `dev` with it. Then check how the "
        f"commit got past the ruleset that restricts updates to `{branch}`.",
    ]


def failed_section(branch: str, source_branch: str, error: SyncError) -> List[str]:
    run = os.environ.get("GITHUB_RUN_ID")
    where = (
        f" ([run]({os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
        f"{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{run}))"
        if run
        else ""
    )
    return [
        f"The docs sync could not update `{branch}`{where}:",
        "",
        f"    {error}",
        "",
        f"The published docs will not follow `{source_branch}` until this is fixed.",
    ]


def report(gh: GitHub, lines: List[str], notify: str, key: str) -> str:
    """File the report on the sync's open issue, or open one.

    Only an issue this account opened is reused: the repository is public, so
    anyone can open one with the same title.

    `key` names what is being reported. The sync runs daily and on every push
    to master, so a problem that persists would otherwise be posted again by
    every run. A report whose key is already on the open issue is not repeated.
    """
    marker = f"<!-- gitbook-sync:{key} -->"
    body = "\n".join(lines + (["", f"cc {notify}"] if notify else []) + ["", marker])
    creator = urllib.parse.quote(gh.actor)
    mine = gh.call("GET", f"/issues?state=open&creator={creator}&per_page=100")
    existing = [
        i for i in mine if i["title"] == ISSUE_TITLE and "pull_request" not in i
    ]
    if not existing:
        created = gh.call(
            "POST", "/issues", {"title": ISSUE_TITLE, "body": body}, retry=False
        )
        return created["html_url"]
    issue = existing[0]
    comments = gh.call("GET", f"/issues/{issue['number']}/comments?per_page=100")
    said = [issue.get("body") or ""] + [c.get("body") or "" for c in comments]
    if any(marker in text for text in said):
        print(f"Already reported on {issue['html_url']}; not repeating it.")
    else:
        gh.call(
            "POST", f"/issues/{issue['number']}/comments", {"body": body}, retry=False
        )
    return issue["html_url"]


def summarize(line: str) -> None:
    print(line)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def sync(
    gh: GitHub, source_branch: str, target_branch: str, notify: str, dry_run: bool
) -> int:
    source = read_snapshot(gh, branch_head(gh, source_branch))
    if not source.docs_tree:
        raise SyncError(
            f"{source_branch}@{source.sha} has no {DOCS_PATH}/ directory to copy"
        )

    target = read_snapshot(gh, branch_head(gh, target_branch))
    own = is_own_snapshot(gh, target)
    if own and target.docs_tree == source.docs_tree:
        summarize(
            f"`{target_branch}` already holds `{DOCS_PATH}/` as of "
            f"`{source_branch}@{source.sha[:10]}`. Nothing to do."
        )
        return 0

    found: List[str] = []
    if not own:
        foreign, reached_snapshot = foreign_commits(gh, target.sha)
        if reached_snapshot or has_written_before(gh, target_branch):
            for commit in foreign[:MAX_REPORTED]:
                print(f"::warning::Not written by the sync: {describe(commit)}")
            found = foreign_section(
                gh, target_branch, source_branch, foreign, reached_snapshot
            )
        else:
            print(f"The sync has never written {target_branch}: first sync.")

    if dry_run:
        summarize(
            f"Dry run: would write `{DOCS_PATH}/` from `{source_branch}@"
            f"{source.sha[:10]}` on top of `{target_branch}@{target.sha[:10]}`"
            + (" and report commits the sync did not write." if found else ".")
        )
        return 0

    if found:
        # Reported before the branch moves. Once the snapshot is on top, the
        # next run sees nothing wrong, so a report that failed after the write
        # would never be sent.
        url = report(gh, found, notify, key=f"foreign:{target.sha}")
        summarize(f"Reported commits the sync did not write: {url}")
    written = write_snapshot(gh, target_branch, target.sha, source_branch, source)
    summarize(
        f"Wrote `{DOCS_PATH}/` from `{source_branch}@{source.sha[:10]}` to "
        f"`{target_branch}` as `{written[:10]}`."
    )
    if found:
        print("Failing the run so that the foreign commits are noticed.")
        return 1
    return 0


def run(
    gh: GitHub, source_branch: str, target_branch: str, notify: str, dry_run: bool
) -> int:
    try:
        return sync(gh, source_branch, target_branch, notify, dry_run)
    except SyncError as e:
        print(f"Error: {e}")
        if dry_run:
            return 1
        # A sync that cannot run leaves the site stale with nobody watching,
        # so say so where the foreign commits are reported.
        try:
            report(
                gh,
                failed_section(target_branch, source_branch, e),
                notify,
                key="failed:" + hashlib.sha1(str(e).encode()).hexdigest()[:12],
            )
        except SyncError as report_error:
            print(f"Could not report the failure: {report_error}")
        return 1


def main() -> int:
    try:
        gh = GitHub(
            os.environ.get("GITHUB_API_URL", "https://api.github.com"),
            os.environ["GITHUB_REPOSITORY"],
            os.environ["GITHUB_TOKEN"],
            os.environ["SYNC_ACTOR"],
        )
    except KeyError as e:
        print(f"Error: Missing required environment variable: {e}")
        return 1
    return run(
        gh,
        source_branch=os.environ.get("SOURCE_BRANCH", "master"),
        target_branch=os.environ.get("TARGET_BRANCH", "gitbook"),
        notify=os.environ.get("NOTIFY", ""),
        dry_run=os.environ.get("DRY_RUN", "").lower() in ("1", "true"),
    )


if __name__ == "__main__":
    sys.exit(main())
