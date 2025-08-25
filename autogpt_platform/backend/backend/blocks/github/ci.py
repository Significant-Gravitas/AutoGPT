import logging
import re
from enum import Enum
from typing import Optional

from typing_extensions import TypedDict

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

from ._api import get_api
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)

logger = logging.getLogger(__name__)


class CheckRunStatus(Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class CheckRunConclusion(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"


class GithubGetCIResultsBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description="GitHub repository",
            placeholder="owner/repo",
        )
        target: str | int = SchemaField(
            description="Commit SHA or PR number to get CI results for",
            placeholder="abc123def or 123",
        )
        search_pattern: Optional[str] = SchemaField(
            description="Optional regex pattern to search for in CI logs (e.g., error messages, file names)",
            placeholder=".*error.*|.*warning.*",
            default=None,
            advanced=True,
        )
        check_name_filter: Optional[str] = SchemaField(
            description="Optional filter for specific check names (supports wildcards)",
            placeholder="*lint* or build-*",
            default=None,
            advanced=True,
        )

    class Output(BlockSchema):
        class CheckRunItem(TypedDict, total=False):
            id: int
            name: str
            status: str
            conclusion: Optional[str]
            started_at: Optional[str]
            completed_at: Optional[str]
            html_url: str
            details_url: Optional[str]
            output_title: Optional[str]
            output_summary: Optional[str]
            output_text: Optional[str]
            annotations: list[dict]

        class MatchedLine(TypedDict):
            check_name: str
            line_number: int
            line: str
            context: list[str]

        check_run: CheckRunItem = SchemaField(
            title="Check Run",
            description="Individual CI check run with details",
        )
        check_runs: list[CheckRunItem] = SchemaField(
            description="List of all CI check runs"
        )
        matched_line: MatchedLine = SchemaField(
            title="Matched Line",
            description="Line matching the search pattern with context",
        )
        matched_lines: list[MatchedLine] = SchemaField(
            description="All lines matching the search pattern across all checks"
        )
        overall_status: str = SchemaField(
            description="Overall CI status (pending, success, failure)"
        )
        overall_conclusion: str = SchemaField(
            description="Overall CI conclusion if completed"
        )
        total_checks: int = SchemaField(description="Total number of CI checks")
        passed_checks: int = SchemaField(description="Number of passed checks")
        failed_checks: int = SchemaField(description="Number of failed checks")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="8ad9e103-78f2-4fdb-ba12-3571f2c95e98",
            description="This block gets CI results for a commit or PR, with optional search for specific errors/warnings in logs.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubGetCIResultsBlock.Input,
            output_schema=GithubGetCIResultsBlock.Output,
            test_input={
                "repo": "owner/repo",
                "target": "abc123def456",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("overall_status", "completed"),
                ("overall_conclusion", "success"),
                ("total_checks", 1),
                ("passed_checks", 1),
                ("failed_checks", 0),
                (
                    "check_runs",
                    [
                        {
                            "id": 123456,
                            "name": "build",
                            "status": "completed",
                            "conclusion": "success",
                            "started_at": "2024-01-01T00:00:00Z",
                            "completed_at": "2024-01-01T00:05:00Z",
                            "html_url": "https://github.com/owner/repo/runs/123456",
                            "details_url": None,
                            "output_title": "Build passed",
                            "output_summary": "All tests passed",
                            "output_text": "Build log output...",
                            "annotations": [],
                        }
                    ],
                ),
            ],
            test_mock={
                "get_ci_results": lambda *args, **kwargs: {
                    "check_runs": [
                        {
                            "id": 123456,
                            "name": "build",
                            "status": "completed",
                            "conclusion": "success",
                            "started_at": "2024-01-01T00:00:00Z",
                            "completed_at": "2024-01-01T00:05:00Z",
                            "html_url": "https://github.com/owner/repo/runs/123456",
                            "details_url": None,
                            "output_title": "Build passed",
                            "output_summary": "All tests passed",
                            "output_text": "Build log output...",
                            "annotations": [],
                        }
                    ],
                    "total_count": 1,
                }
            },
        )

    @staticmethod
    async def get_commit_sha(api, repo: str, target: str | int) -> str:
        """Get commit SHA from either a commit SHA or PR URL."""
        # If it's already a SHA, return it

        if isinstance(target, str):
            if re.match(r"^[0-9a-f]{6,40}$", target, re.IGNORECASE):
                return target

        # If it's a PR URL, get the head SHA
        if isinstance(target, int):
            pr_url = f"https://api.github.com/repos/{repo}/pulls/{target}"
            response = await api.get(pr_url)
            pr_data = response.json()
            return pr_data["head"]["sha"]

        raise ValueError("Target must be a commit SHA or PR URL")

    @staticmethod
    async def search_in_logs(
        check_runs: list,
        pattern: str,
    ) -> list[Output.MatchedLine]:
        """Search for pattern in check run logs."""
        if not pattern:
            return []

        matched_lines = []
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

        for check in check_runs:
            output_text = check.get("output_text", "") or ""
            if not output_text:
                continue

            lines = output_text.split("\n")
            for i, line in enumerate(lines):
                if regex.search(line):
                    # Get context (2 lines before and after)
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context = lines[start:end]

                    matched_lines.append(
                        {
                            "check_name": check["name"],
                            "line_number": i + 1,
                            "line": line,
                            "context": context,
                        }
                    )

        return matched_lines

    @staticmethod
    async def get_ci_results(
        credentials: GithubCredentials,
        repo: str,
        target: str | int,
        search_pattern: Optional[str] = None,
        check_name_filter: Optional[str] = None,
    ) -> dict:
        api = get_api(credentials, convert_urls=False)

        # Get the commit SHA
        commit_sha = await GithubGetCIResultsBlock.get_commit_sha(api, repo, target)

        # Get check runs for the commit
        check_runs_url = (
            f"https://api.github.com/repos/{repo}/commits/{commit_sha}/check-runs"
        )

        # Get all pages of check runs
        all_check_runs = []
        page = 1
        per_page = 100

        while True:
            response = await api.get(
                check_runs_url, params={"per_page": per_page, "page": page}
            )
            data = response.json()

            check_runs = data.get("check_runs", [])
            all_check_runs.extend(check_runs)

            if len(check_runs) < per_page:
                break
            page += 1

        # Filter by check name if specified
        if check_name_filter:
            import fnmatch

            filtered_runs = []
            for run in all_check_runs:
                if fnmatch.fnmatch(run["name"].lower(), check_name_filter.lower()):
                    filtered_runs.append(run)
            all_check_runs = filtered_runs

        # Get check run details with logs
        detailed_runs = []
        for run in all_check_runs:
            # Get detailed output including logs
            if run.get("output", {}).get("text"):
                # Already has output
                detailed_run = {
                    "id": run["id"],
                    "name": run["name"],
                    "status": run["status"],
                    "conclusion": run.get("conclusion"),
                    "started_at": run.get("started_at"),
                    "completed_at": run.get("completed_at"),
                    "html_url": run["html_url"],
                    "details_url": run.get("details_url"),
                    "output_title": run.get("output", {}).get("title"),
                    "output_summary": run.get("output", {}).get("summary"),
                    "output_text": run.get("output", {}).get("text"),
                    "annotations": [],
                }
            else:
                # Try to get logs from the check run
                detailed_run = {
                    "id": run["id"],
                    "name": run["name"],
                    "status": run["status"],
                    "conclusion": run.get("conclusion"),
                    "started_at": run.get("started_at"),
                    "completed_at": run.get("completed_at"),
                    "html_url": run["html_url"],
                    "details_url": run.get("details_url"),
                    "output_title": run.get("output", {}).get("title"),
                    "output_summary": run.get("output", {}).get("summary"),
                    "output_text": None,
                    "annotations": [],
                }

            # Get annotations if available
            if run.get("output", {}).get("annotations_count", 0) > 0:
                annotations_url = f"https://api.github.com/repos/{repo}/check-runs/{run['id']}/annotations"
                try:
                    ann_response = await api.get(annotations_url)
                    detailed_run["annotations"] = ann_response.json()
                except Exception:
                    pass

            detailed_runs.append(detailed_run)

        return {
            "check_runs": detailed_runs,
            "total_count": len(detailed_runs),
        }

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:

        try:
            target = int(input_data.target)
        except ValueError:
            target = input_data.target

        result = await self.get_ci_results(
            credentials,
            input_data.repo,
            target,
            input_data.search_pattern,
            input_data.check_name_filter,
        )

        check_runs = result["check_runs"]

        # Calculate overall status
        if not check_runs:
            yield "overall_status", "no_checks"
            yield "overall_conclusion", "no_checks"
        else:
            all_completed = all(run["status"] == "completed" for run in check_runs)
            if all_completed:
                yield "overall_status", "completed"
                # Determine overall conclusion
                has_failure = any(
                    run["conclusion"] in ["failure", "timed_out", "action_required"]
                    for run in check_runs
                )
                if has_failure:
                    yield "overall_conclusion", "failure"
                else:
                    yield "overall_conclusion", "success"
            else:
                yield "overall_status", "pending"
                yield "overall_conclusion", "pending"

        # Count checks
        total = len(check_runs)
        passed = sum(1 for run in check_runs if run.get("conclusion") == "success")
        failed = sum(
            1 for run in check_runs if run.get("conclusion") in ["failure", "timed_out"]
        )

        yield "total_checks", total
        yield "passed_checks", passed
        yield "failed_checks", failed

        # Output check runs
        yield "check_runs", check_runs

        # Search for patterns if specified
        if input_data.search_pattern:
            matched_lines = await self.search_in_logs(
                check_runs, input_data.search_pattern
            )
            if matched_lines:
                yield "matched_lines", matched_lines
