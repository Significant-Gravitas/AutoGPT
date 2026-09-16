import json
import os
import requests
import sys
import time
from typing import Dict, List, Optional, Tuple

CHECK_INTERVAL = 30
PASSING_CONCLUSIONS = {"success", "skipped", "neutral"}


def get_environment_variables() -> Tuple[str, str, str, str, str]:
    """Retrieve and return necessary environment variables."""
    try:
        with open(os.environ["GITHUB_EVENT_PATH"]) as f:
            event = json.load(f)

        # Handle both PR and merge group events
        if "pull_request" in event:
            sha = event["pull_request"]["head"]["sha"]
        else:
            sha = os.environ["GITHUB_SHA"]

        return (
            os.environ["GITHUB_API_URL"],
            os.environ["GITHUB_REPOSITORY"],
            sha,
            os.environ["GITHUB_TOKEN"],
            os.environ["GITHUB_RUN_ID"],
        )
    except KeyError as e:
        print(f"Error: Missing required environment variable or event data: {e}")
        sys.exit(1)


def make_api_request(
    url: str, headers: Dict[str, str]
) -> Tuple[Dict, Optional[str]]:
    """Make an API request and return its JSON and next-page URL."""
    try:
        print("Making API request to:", url)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        next_url = response.links.get("next", {}).get("url")
        return response.json(), next_url
    except requests.RequestException as e:
        print(f"Error: API request failed. {e}")
        sys.exit(1)


def get_paginated_items(
    url: str, headers: Dict[str, str], item_key: str
) -> List[Dict]:
    """Retrieve all pages from a GitHub list endpoint."""
    items = []

    while url:
        data, url = make_api_request(url, headers)
        items.extend(data[item_key])

    return items


def get_action_run_id(check_run: Dict) -> Optional[int]:
    """Extract a GitHub Actions run ID from a check run details URL."""
    details_url = str(check_run.get("details_url", ""))
    marker = "/actions/runs/"
    if marker not in details_url:
        return None

    run_id = details_url.split(marker, 1)[1].split("/", 1)[0]
    try:
        return int(run_id)
    except ValueError:
        return None


def action_run_order(run: Dict) -> Tuple[int, int, int]:
    """Return an ordering key that also handles rerun attempts."""
    return (
        int(run.get("run_number", 0)),
        int(run.get("run_attempt", 0)),
        int(run.get("id", 0)),
    )


def is_superseded_cancellation(
    check_run: Dict, workflow_runs: List[Dict]
) -> bool:
    """Return whether a canceled check belongs to a superseded workflow run."""
    action_run_id = get_action_run_id(check_run)
    if action_run_id is None:
        return False

    runs_by_id = {int(run["id"]): run for run in workflow_runs}
    source_run = runs_by_id.get(action_run_id)
    if source_run is None:
        return False

    for candidate in workflow_runs:
        if candidate.get("workflow_id") != source_run.get("workflow_id"):
            continue
        if candidate.get("head_sha") != source_run.get("head_sha"):
            continue
        if candidate.get("event") != source_run.get("event"):
            continue
        if action_run_order(candidate) <= action_run_order(source_run):
            continue
        if (
            candidate.get("status") == "completed"
            and candidate.get("conclusion") in PASSING_CONCLUSIONS
        ):
            return True

    return False


def process_check_runs(
    check_runs: List[Dict], workflow_runs: Optional[List[Dict]] = None
) -> Tuple[bool, bool]:
    """Process check runs and return their status."""
    workflow_runs = workflow_runs or []
    runs_in_progress = False
    all_others_passed = True

    for run in check_runs:
        if str(run["name"]) != "Check PR Status":
            status = run["status"]
            conclusion = run["conclusion"]

            if status == "completed":
                if conclusion not in PASSING_CONCLUSIONS:
                    if conclusion == "cancelled" and is_superseded_cancellation(
                        run, workflow_runs
                    ):
                        print(
                            f"Ignoring canceled check run {run['name']} "
                            f"(ID: {run['id']}) because a newer run of the "
                            "same workflow passed."
                        )
                        continue
                    all_others_passed = False
                    print(
                        f"Check run {run['name']} (ID: {run['id']}) has conclusion: {conclusion}"
                    )
            else:
                runs_in_progress = True
                print(f"Check run {run['name']} (ID: {run['id']}) is still {status}.")
                all_others_passed = False
        else:
            print(
                f"Skipping check run {run['name']} (ID: {run['id']}) as it is the current run."
            )

    return runs_in_progress, all_others_passed


def main():
    api_url, repo, sha, github_token, current_run_id = get_environment_variables()

    check_runs_endpoint = (
        f"{api_url}/repos/{repo}/commits/{sha}/check-runs?per_page=100"
    )
    workflow_runs_endpoint = (
        f"{api_url}/repos/{repo}/actions/runs?head_sha={sha}&per_page=100"
    )
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    print(f"Current run ID: {current_run_id}")

    while True:
        check_runs = get_paginated_items(
            check_runs_endpoint, headers, "check_runs"
        )
        print(f"Processing {len(check_runs)} check runs...")

        runs_in_progress = any(
            str(run["name"]) != "Check PR Status"
            and run["status"] != "completed"
            for run in check_runs
        )

        if not runs_in_progress:
            workflow_runs = get_paginated_items(
                workflow_runs_endpoint, headers, "workflow_runs"
            )
            _, all_others_passed = process_check_runs(
                check_runs, workflow_runs
            )
            break

        print(
            "Some check runs are still in progress. "
            f"Waiting {CHECK_INTERVAL} seconds before checking again..."
        )
        time.sleep(CHECK_INTERVAL)

    if all_others_passed:
        print("All other completed check runs have passed. This check passes.")
        sys.exit(0)
    else:
        print("Some check runs have failed or have not completed. This check fails.")
        sys.exit(1)


if __name__ == "__main__":
    main()
