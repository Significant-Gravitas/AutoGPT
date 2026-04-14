"""
GitHub Issues Block — fetches issues from a GitHub repository and converts them
into agent tasks. Supports filtering by label, assignee, milestone, and state.

Enables the agent to automatically pick up open issues and work on them.
"""

import logging
from enum import Enum
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


class IssueOperation(str, Enum):
    LIST = "list"
    GET = "get"
    CREATE_TASK = "create_task"
    COMMENT = "comment"
    CLOSE = "close"


class GitHubIssuesInput(BlockSchemaInput):
    operation: IssueOperation = SchemaField(
        default=IssueOperation.LIST,
        description="Operation: list issues, get a specific issue, create task from issue, comment, or close.",
    )
    github_token: str = SchemaField(
        default="",
        description="GitHub Personal Access Token (PAT) with repo scope.",
    )
    repo_owner: str = SchemaField(
        description="GitHub repository owner (username or org).",
    )
    repo_name: str = SchemaField(
        description="GitHub repository name.",
    )
    issue_number: int = SchemaField(
        default=0,
        description="Issue number (for GET, COMMENT, CLOSE, CREATE_TASK operations).",
    )
    state: str = SchemaField(
        default="open",
        description="Issue state filter: 'open', 'closed', or 'all'.",
    )
    labels: list = SchemaField(
        default_factory=list,
        description="Filter issues by labels (e.g., ['bug', 'enhancement']).",
    )
    assignee: str = SchemaField(
        default="",
        description="Filter issues by assignee username.",
    )
    max_issues: int = SchemaField(
        default=10,
        description="Maximum number of issues to fetch.",
    )
    comment_body: str = SchemaField(
        default="",
        description="Comment text (for COMMENT operation).",
    )


class GitHubIssuesOutput(BlockSchemaOutput):
    issues: list = SchemaField(description="List of issue objects.")
    issue_title: str = SchemaField(description="Title of the fetched/targeted issue.")
    issue_body: str = SchemaField(description="Body of the fetched/targeted issue.")
    issue_url: str = SchemaField(description="URL of the issue.")
    task_prompt: str = SchemaField(description="Agent task prompt generated from the issue.")
    count: int = SchemaField(description="Number of issues fetched.")
    status: str = SchemaField(description="Operation result status.")


class GitHubIssuesBlock(Block):
    """
    Fetches GitHub issues and converts them into coding agent tasks.

    List open issues, filter by label/assignee, and automatically generate
    task prompts for the agent to work on. Supports commenting and closing
    issues when work is complete.
    """

    class Input(GitHubIssuesInput):
        pass

    class Output(GitHubIssuesOutput):
        pass

    def __init__(self):
        super().__init__(
            id="a7b8c9d0-e1f2-3456-abcd-789012345678",
            description=(
                "Fetches GitHub issues and converts them into agent tasks. "
                "Filter by label, assignee, or state. Auto-generates task prompts."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GitHubIssuesBlock.Input,
            output_schema=GitHubIssuesBlock.Output,
            test_input={
                "operation": IssueOperation.LIST.value,
                "github_token": "test_token",
                "repo_owner": "test_owner",
                "repo_name": "test_repo",
                "state": "open",
                "max_issues": 5,
            },
            test_output=[
                ("count", 0),
                ("status", "No GitHub token provided or API call failed."),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        try:
            import requests
        except ImportError:
            yield "issues", []
            yield "issue_title", ""
            yield "issue_body", ""
            yield "issue_url", ""
            yield "task_prompt", ""
            yield "count", 0
            yield "status", "requests library not available."
            return

        token = input_data.github_token
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        base_url = f"https://api.github.com/repos/{input_data.repo_owner}/{input_data.repo_name}"

        if input_data.operation == IssueOperation.LIST:
            params = {
                "state": input_data.state,
                "per_page": min(input_data.max_issues, 100),
            }
            if input_data.labels:
                params["labels"] = ",".join(input_data.labels)
            if input_data.assignee:
                params["assignee"] = input_data.assignee

            try:
                resp = requests.get(f"{base_url}/issues", headers=headers, params=params, timeout=15)
                if resp.status_code == 200:
                    issues = resp.json()
                    # Filter out PRs (GitHub API returns PRs in issues endpoint)
                    issues = [i for i in issues if "pull_request" not in i]
                    simplified = [
                        {
                            "number": i["number"],
                            "title": i["title"],
                            "state": i["state"],
                            "labels": [l["name"] for l in i.get("labels", [])],
                            "url": i["html_url"],
                            "body": (i.get("body") or "")[:500],
                        }
                        for i in issues
                    ]
                    yield "issues", simplified
                    yield "issue_title", simplified[0]["title"] if simplified else ""
                    yield "issue_body", simplified[0]["body"] if simplified else ""
                    yield "issue_url", simplified[0]["url"] if simplified else ""
                    yield "task_prompt", ""
                    yield "count", len(simplified)
                    yield "status", f"Fetched {len(simplified)} issues."
                else:
                    yield "issues", []
                    yield "issue_title", ""
                    yield "issue_body", ""
                    yield "issue_url", ""
                    yield "task_prompt", ""
                    yield "count", 0
                    yield "status", f"GitHub API error {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                yield "issues", []
                yield "issue_title", ""
                yield "issue_body", ""
                yield "issue_url", ""
                yield "task_prompt", ""
                yield "count", 0
                yield "status", f"Request failed: {e}"

        elif input_data.operation in (IssueOperation.GET, IssueOperation.CREATE_TASK):
            if not input_data.issue_number:
                yield "issues", []
                yield "issue_title", ""
                yield "issue_body", ""
                yield "issue_url", ""
                yield "task_prompt", ""
                yield "count", 0
                yield "status", "issue_number is required for GET/CREATE_TASK."
                return
            try:
                resp = requests.get(
                    f"{base_url}/issues/{input_data.issue_number}",
                    headers=headers, timeout=15,
                )
                if resp.status_code == 200:
                    issue = resp.json()
                    title = issue["title"]
                    body = issue.get("body") or ""
                    url = issue["html_url"]
                    labels = [l["name"] for l in issue.get("labels", [])]

                    task_prompt = (
                        f"GitHub Issue #{input_data.issue_number}: {title}\n\n"
                        f"Repository: {input_data.repo_owner}/{input_data.repo_name}\n"
                        f"Labels: {', '.join(labels) if labels else 'none'}\n"
                        f"URL: {url}\n\n"
                        f"## Issue Description\n{body}\n\n"
                        f"## Task\nAnalyze the issue above and implement a fix or feature. "
                        f"Write clean, tested code following the project's existing patterns. "
                        f"After completing the implementation, summarize what was done."
                    )

                    yield "issues", [{"number": issue["number"], "title": title, "url": url}]
                    yield "issue_title", title
                    yield "issue_body", body
                    yield "issue_url", url
                    yield "task_prompt", task_prompt
                    yield "count", 1
                    yield "status", f"Issue #{input_data.issue_number} fetched."
                else:
                    yield "issues", []
                    yield "issue_title", ""
                    yield "issue_body", ""
                    yield "issue_url", ""
                    yield "task_prompt", ""
                    yield "count", 0
                    yield "status", f"GitHub API error {resp.status_code}"
            except Exception as e:
                yield "issues", []
                yield "issue_title", ""
                yield "issue_body", ""
                yield "issue_url", ""
                yield "task_prompt", ""
                yield "count", 0
                yield "status", f"Request failed: {e}"

        elif input_data.operation == IssueOperation.COMMENT:
            if not input_data.issue_number or not input_data.comment_body:
                yield "issues", []
                yield "issue_title", ""
                yield "issue_body", ""
                yield "issue_url", ""
                yield "task_prompt", ""
                yield "count", 0
                yield "status", "issue_number and comment_body required for COMMENT."
                return
            try:
                resp = requests.post(
                    f"{base_url}/issues/{input_data.issue_number}/comments",
                    headers=headers,
                    json={"body": input_data.comment_body},
                    timeout=15,
                )
                yield "issues", []
                yield "issue_title", ""
                yield "issue_body", ""
                yield "issue_url", f"{base_url}/issues/{input_data.issue_number}"
                yield "task_prompt", ""
                yield "count", 1 if resp.status_code == 201 else 0
                yield "status", (
                    f"Comment posted." if resp.status_code == 201
                    else f"Failed: {resp.status_code}"
                )
            except Exception as e:
                yield "issues", []
                yield "issue_title", ""
                yield "issue_body", ""
                yield "issue_url", ""
                yield "task_prompt", ""
                yield "count", 0
                yield "status", f"Request failed: {e}"

        elif input_data.operation == IssueOperation.CLOSE:
            if not input_data.issue_number:
                yield "issues", []
                yield "issue_title", ""
                yield "issue_body", ""
                yield "issue_url", ""
                yield "task_prompt", ""
                yield "count", 0
                yield "status", "issue_number required for CLOSE."
                return
            try:
                resp = requests.patch(
                    f"{base_url}/issues/{input_data.issue_number}",
                    headers=headers,
                    json={"state": "closed"},
                    timeout=15,
                )
                yield "issues", []
                yield "issue_title", ""
                yield "issue_body", ""
                yield "issue_url", f"{base_url}/issues/{input_data.issue_number}"
                yield "task_prompt", ""
                yield "count", 1 if resp.status_code == 200 else 0
                yield "status", (
                    f"Issue #{input_data.issue_number} closed."
                    if resp.status_code == 200
                    else f"Failed: {resp.status_code}"
                )
            except Exception as e:
                yield "issues", []
                yield "issue_title", ""
                yield "issue_body", ""
                yield "issue_url", ""
                yield "task_prompt", ""
                yield "count", 0
                yield "status", f"Request failed: {e}"
        else:
            yield "issues", []
            yield "issue_title", ""
            yield "issue_body", ""
            yield "issue_url", ""
            yield "task_prompt", ""
            yield "count", 0
            yield "status", f"No GitHub token provided or API call failed."
