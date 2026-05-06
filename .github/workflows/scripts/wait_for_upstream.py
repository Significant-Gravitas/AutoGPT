"""Wait for upstream workflow runs (by display name) on the given head SHA.

Used by platform-fullstack-ci's preflight job to gate the expensive
big-boi E2E job: only run E2E if every listed upstream workflow has
completed successfully on the same SHA. Outputs ``proceed=true|false``
to ``$GITHUB_OUTPUT`` for downstream ``if:`` conditions.

A workflow that has not appeared on the SHA after the grace period is
treated as not-triggered and blocks the gate (proceed=false), so a slow
webhook or path-filtered upstream cannot let the expensive job through.
"""

import os
import sys
import time
from typing import Dict, List, Optional

import requests

REQUEST_TIMEOUT = 15
# Wait this long before we even start polling — gives webhooks +
# workflow registration time on busy queues.
INITIAL_DELAY = 60
POLL_INTERVAL = 30
MAX_WAIT_SECONDS = 90 * 60
# Grace period (measured from preflight start) during which a missing
# upstream workflow is treated as "still pending registration" rather
# than "path-filtered out / never going to run". Without this, a slow
# webhook delivery could let us emit proceed=true before backend/frontend
# CI have even appeared on the SHA — defeating the cost guard.
MISSING_GRACE_SECONDS = 10 * 60


def get_env() -> Optional[Dict[str, object]]:
    raw = os.environ.get("UPSTREAM_WORKFLOWS", "")
    workflows = [w.strip() for w in raw.splitlines() if w.strip()]
    if not workflows:
        return None
    return {
        "api": os.environ["GITHUB_API_URL"],
        "repo": os.environ["GITHUB_REPOSITORY"],
        "sha": os.environ["HEAD_SHA"],
        "token": os.environ["GITHUB_TOKEN"],
        "workflows": workflows,
    }


def fetch_runs(env: Dict[str, object], headers: Dict[str, str]) -> List[Dict]:
    runs: List[Dict] = []
    url = (
        f"{env['api']}/repos/{env['repo']}/actions/runs"
        f"?head_sha={env['sha']}&per_page=100"
    )
    while url:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        runs.extend(response.json().get("workflow_runs", []))
        next_url = None
        for part in response.headers.get("Link", "").split(","):
            if 'rel="next"' in part:
                next_url = part.split(";")[0].strip().strip("<>")
                break
        url = next_url
    return runs


def latest_per_workflow(runs: List[Dict], names: List[str]) -> Dict[str, Dict]:
    by_name: Dict[str, Dict] = {}
    name_set = set(names)
    for run in runs:
        name = run.get("name")
        if name not in name_set:
            continue
        prev = by_name.get(name)
        run_key = (run["run_number"], run.get("run_attempt", 0))
        if prev is None or run_key > (
            prev["run_number"],
            prev.get("run_attempt", 0),
        ):
            by_name[name] = run
    return by_name


def write_output(key: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"{key}={value}\n")
    print(f"output: {key}={value}")


def main() -> None:
    env = get_env()
    if env is None:
        print("No UPSTREAM_WORKFLOWS configured; proceeding.")
        write_output("proceed", "true")
        return

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {env['token']}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    print(f"Gating on upstream workflows for SHA {env['sha']}:")
    for w in env["workflows"]:
        print(f"  - {w}")
    start = time.monotonic()
    print(f"Initial delay {INITIAL_DELAY}s to let upstream runs register...")
    time.sleep(INITIAL_DELAY)

    deadline = start + MAX_WAIT_SECONDS
    grace_deadline = start + MISSING_GRACE_SECONDS

    while True:
        try:
            runs = fetch_runs(env, headers)
        except requests.RequestException as exc:
            if time.monotonic() > deadline:
                print(
                    f"Timeout after {MAX_WAIT_SECONDS}s while fetching runs: {exc}"
                )
                write_output("proceed", "false")
                sys.exit(1)
            print(f"Transient GitHub API error while fetching runs: {exc}")
            time.sleep(POLL_INTERVAL)
            continue

        by_name = latest_per_workflow(runs, env["workflows"])

        missing = [w for w in env["workflows"] if w not in by_name]
        in_progress = [
            w for w, r in by_name.items() if r["status"] != "completed"
        ]
        failed = [
            f"{w}={by_name[w]['conclusion']}"
            for w in by_name
            if by_name[w]["status"] == "completed"
            and by_name[w]["conclusion"] not in ("success", "skipped", "neutral")
        ]

        if failed:
            print(f"Upstream failed: {failed}")
            write_output("proceed", "false")
            return

        # Don't treat "missing" as "skipped" until the grace period
        # has fully elapsed — otherwise a slow webhook / queue could
        # let us pass the gate before backend/frontend CI even appears.
        if missing and time.monotonic() < grace_deadline:
            remaining = int(grace_deadline - time.monotonic())
            print(
                f"Waiting (within {MISSING_GRACE_SECONDS}s grace, "
                f"{remaining}s left): missing={missing}, "
                f"in_progress={in_progress}"
            )
            time.sleep(POLL_INTERVAL)
            continue

        if not in_progress:
            if missing:
                print(
                    "Workflow(s) did not trigger after grace period "
                    f"(treating as not-triggered, blocking): {missing}"
                )
                write_output("proceed", "false")
                return
            print(f"Upstream passed: {list(by_name.keys())}")
            write_output("proceed", "true")
            return

        if time.monotonic() > deadline:
            print(
                f"Timeout after {MAX_WAIT_SECONDS}s; still in_progress="
                f"{in_progress}, missing={missing}"
            )
            write_output("proceed", "false")
            sys.exit(1)

        print(f"Waiting: in_progress={in_progress}, missing={missing}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
