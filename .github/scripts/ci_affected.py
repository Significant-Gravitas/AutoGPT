import argparse
import json
import os
import re
import subprocess
from pathlib import Path

PROJECTS = {"backend", "frontend", "fullstack"}
ROOT = Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8")


def revisions(event_name: str, event: dict) -> tuple[str, str] | None:
    if event_name in {"schedule", "workflow_dispatch"}:
        return None
    if event_name == "pull_request":
        base = event["pull_request"]["base"]["sha"]
        head = event["pull_request"]["head"]["sha"]
    elif event_name == "merge_group":
        base = event["merge_group"]["base_sha"]
        head = event["merge_group"]["head_sha"]
    else:
        raise ValueError(f"Unsupported CI selection event: {event_name}")
    if not all(re.fullmatch(r"[0-9a-f]{40,64}", sha) for sha in (base, head)):
        raise ValueError("Expected complete base and head commit IDs")
    if event_name == "pull_request":
        base = git("merge-base", base, head).strip()
    return base, head


def matches(path: str, prefix: str) -> bool:
    return path.startswith(prefix) if prefix.endswith("/") else path == prefix


def select_projects(files: list[str], config: dict) -> set[str]:
    selected = set()
    for path in files:
        owners = {
            project
            for project in PROJECTS
            if any(matches(path, prefix) for prefix in config[project])
        }
        if owners == {"fullstack"}:
            selected.update(PROJECTS)
        elif owners:
            selected.update(owners)
        elif not any(matches(path, prefix) for prefix in config["unrelated"]):
            selected.update(PROJECTS)
    return selected


def required_projects(files: list[str]) -> set[str]:
    """Independent safety floor: never consult the selector's graph or file list."""
    required = set()
    for path in files:
        if path.startswith(("docs/", "classic/", ".github/ISSUE_TEMPLATE/")):
            continue
        if not path.startswith("autogpt_platform/"):
            required.update(PROJECTS)
            continue
        required.add("fullstack")
        if path.startswith(
            ("autogpt_platform/backend/", "autogpt_platform/autogpt_libs/")
        ):
            required.add("backend")
            if path in {
                "autogpt_platform/backend/schema.prisma",
                "autogpt_platform/backend/Dockerfile",
            }:
                required.add("frontend")
        elif path.startswith("autogpt_platform/frontend/"):
            required.add("frontend")
            if path.startswith("autogpt_platform/frontend/public/integrations/"):
                required.add("backend")
        else:
            required.update(PROJECTS)
    return required


def verify(plan: dict, bounds: tuple[str, str] | None) -> None:
    if set(plan) != PROJECTS or any(type(value) is not bool for value in plan.values()):
        raise ValueError("Selection must name every project with a boolean")
    if bounds is None:
        expected = PROJECTS
    else:
        # Different Git command and independent policy, not the selector's output.
        raw = git(
            "diff-tree",
            "--no-commit-id",
            "-r",
            "--no-renames",
            "--name-only",
            "-z",
            *bounds,
        )
        expected = required_projects([path for path in raw.split("\0") if path])
    missing = sorted(project for project in expected if not plan[project])
    if missing:
        raise ValueError(f"Affected projects incorrectly skipped: {', '.join(missing)}")


def gate(plan: dict, results: dict) -> None:
    if results.get("changes", {}).get("result") != "success":
        raise ValueError("Affected-project selection or its independent guard failed")
    for project in PROJECTS:
        result = results.get(project, {}).get("result")
        accepted = {"success"} if plan[project] else {"success", "skipped"}
        if result not in accepted:
            raise ValueError(f"{project}: required={plan[project]}, result={result!r}")


def require_jobs(names: list[str], results: dict) -> None:
    for name in names:
        if results.get(name, {}).get("result") != "success":
            raise ValueError(f"Required job {name} did not succeed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "verify", "gate", "jobs"])
    parser.add_argument("--required-jobs", nargs="+")
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--config", type=Path, default=ROOT / ".github/ci-projects.json"
    )
    args = parser.parse_args()
    if args.mode == "jobs":
        if not args.required_jobs:
            parser.error("jobs requires --required-jobs")
        require_jobs(args.required_jobs, json.loads(os.environ["CI_RESULTS"]))
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    bounds = revisions(os.environ["GITHUB_EVENT_NAME"], event)
    if args.mode == "plan":
        if not args.plan:
            parser.error("plan requires --plan")
        files = (
            []
            if bounds is None
            else [
                path
                for path in git(
                    "diff", "--no-renames", "--name-only", "-z", *bounds
                ).split("\0")
                if path
            ]
        )
        selected = (
            PROJECTS
            if bounds is None
            else select_projects(files, json.loads(args.config.read_text()))
        )
        args.plan.write_text(
            json.dumps({project: project in selected for project in sorted(PROJECTS)})
            + "\n"
        )
    else:
        plan = json.loads(
            args.plan.read_text() if args.plan else os.environ["CI_SELECTION"]
        )
        verify(plan, bounds)
        if args.mode == "gate":
            gate(plan, json.loads(os.environ["CI_RESULTS"]))
        elif output := os.environ.get("GITHUB_OUTPUT"):
            with open(output, "a", encoding="utf-8") as stream:
                stream.write(f"selection={json.dumps(plan)}\n")
                for project, selected in plan.items():
                    stream.write(f"{project}={str(selected).lower()}\n")
        print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
