import requests

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)


class GithubCommentBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        issue_url: str = SchemaField(
            description="URL of the GitHub issue or pull request",
            placeholder="https://github.com/owner/repo/issues/1",
        )
        comment: str = SchemaField(
            description="Comment to post on the issue or pull request",
            placeholder="Enter your comment",
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Status of the comment posting operation")
        error: str = SchemaField(
            description="Error message if the comment posting failed"
        )

    def __init__(self):
        super().__init__(
            id="0001c3d4-5678-90ef-1234-567890abcdef",
            description="This block posts a comment on a specified GitHub issue or pull request using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCommentBlock.Input,
            output_schema=GithubCommentBlock.Output,
            test_input={
                "issue_url": "https://github.com/owner/repo/issues/1",
                "comment": "This is a test comment.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Comment posted successfully")],
            test_mock={
                "post_comment": lambda *args, **kwargs: "Comment posted successfully"
            },
        )

    @staticmethod
    def post_comment(
        credentials: GithubCredentials, issue_url: str, comment: str
    ) -> str:
        try:
            if "/pull/" in issue_url:
                api_url = (
                    issue_url.replace("github.com", "api.github.com/repos").replace(
                        "/pull/", "/issues/"
                    )
                    + "/comments"
                )
            else:
                api_url = (
                    issue_url.replace("github.com", "api.github.com/repos")
                    + "/comments"
                )

            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }
            data = {"body": comment}

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            return "Comment posted successfully"
        except Exception as e:
            return f"Failed to post comment: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.post_comment(
            credentials,
            input_data.issue_url,
            input_data.comment,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status


class GithubMakeIssueBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        title: str = SchemaField(
            description="Title of the issue", placeholder="Enter the issue title"
        )
        body: str = SchemaField(
            description="Body of the issue", placeholder="Enter the issue body"
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Status of the issue creation operation")
        error: str = SchemaField(
            description="Error message if the issue creation failed"
        )

    def __init__(self):
        super().__init__(
            id="0002d3e4-5678-90ab-1234-567890abcdef",
            description="This block creates a new issue on a specified GitHub repository using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubMakeIssueBlock.Input,
            output_schema=GithubMakeIssueBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "title": "Test Issue",
                "body": "This is a test issue.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Issue created successfully")],
            test_mock={
                "create_issue": lambda *args, **kwargs: "Issue created successfully"
            },
        )

    @staticmethod
    def create_issue(
        credentials: GithubCredentials, repo_url: str, title: str, body: str
    ) -> str:
        try:
            api_url = repo_url.replace("github.com", "api.github.com/repos") + "/issues"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }
            data = {"title": title, "body": body}

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            return "Issue created successfully"
        except Exception as e:
            return f"Failed to create issue: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.create_issue(
            credentials,
            input_data.repo_url,
            input_data.title,
            input_data.body,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status


class GithubReadIssueBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        issue_url: str = SchemaField(
            description="URL of the GitHub issue",
            placeholder="https://github.com/owner/repo/issues/1",
        )

    class Output(BlockSchema):
        title: str = SchemaField(description="Title of the issue")
        body: str = SchemaField(description="Body of the issue")
        user: str = SchemaField(description="User who created the issue")
        error: str = SchemaField(
            description="Error message if reading the issue failed"
        )

    def __init__(self):
        super().__init__(
            id="0004e3f4-5678-90ab-1234-567890abcdef",
            description="This block reads the body, title, and user of a specified GitHub issue using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadIssueBlock.Input,
            output_schema=GithubReadIssueBlock.Output,
            test_input={
                "issue_url": "https://github.com/owner/repo/issues/1",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("title", "Title of the issue"),
                ("body", "This is the body of the issue."),
                ("user", "username"),
            ],
            test_mock={
                "read_issue": lambda *args, **kwargs: (
                    "Title of the issue",
                    "This is the body of the issue.",
                    "username",
                )
            },
        )

    @staticmethod
    def read_issue(
        credentials: GithubCredentials, issue_url: str
    ) -> tuple[str, str, str]:
        try:
            api_url = issue_url.replace("github.com", "api.github.com/repos")

            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            title = data.get("title", "No title found")
            body = data.get("body", "No body content found")
            user = data.get("user", {}).get("login", "No user found")

            return title, body, user
        except Exception as e:
            return f"Failed to read issue: {str(e)}", "", ""

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        title, body, user = self.read_issue(
            credentials,
            input_data.issue_url,
        )
        if "Failed" in title:
            yield "error", title
        else:
            yield "title", title
            yield "body", body
            yield "user", user


class GithubListIssuesBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        issues: list[dict[str, str]] = SchemaField(
            description="List of issues with their URLs"
        )
        error: str = SchemaField(description="Error message if listing issues failed")

    def __init__(self):
        super().__init__(
            id="0006h3i4-5678-90ab-1234-567890abcdef",
            description="This block lists all issues for a specified GitHub repository using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListIssuesBlock.Input,
            output_schema=GithubListIssuesBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "issues",
                    [
                        {
                            "title": "Issue 1",
                            "url": "https://github.com/owner/repo/issues/1",
                        }
                    ],
                )
            ],
            test_mock={
                "list_issues": lambda *args, **kwargs: [
                    {
                        "title": "Issue 1",
                        "url": "https://github.com/owner/repo/issues/1",
                    }
                ]
            },
        )

    @staticmethod
    def list_issues(
        credentials: GithubCredentials, repo_url: str
    ) -> list[dict[str, str]]:
        try:
            api_url = repo_url.replace("github.com", "api.github.com/repos") + "/issues"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            issues = [
                {"title": issue["title"], "url": issue["html_url"]} for issue in data
            ]

            return issues
        except Exception as e:
            return [{"title": "Error", "url": f"Failed to list issues: {str(e)}"}]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        issues = self.list_issues(
            credentials,
            input_data.repo_url,
        )
        if any("Failed" in issue["url"] for issue in issues):
            yield "error", issues[0]["url"]
        else:
            yield "issues", issues


class GithubAddLabelBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        issue_url: str = SchemaField(
            description="URL of the GitHub issue or pull request",
            placeholder="https://github.com/owner/repo/issues/1",
        )
        label: str = SchemaField(
            description="Label to add to the issue or pull request",
            placeholder="Enter the label",
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Status of the label addition operation")
        error: str = SchemaField(
            description="Error message if the label addition failed"
        )

    def __init__(self):
        super().__init__(
            id="0011l3m4-5678-90ab-1234-567890abcdef",
            description="This block adds a label to a specified GitHub issue or pull request using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubAddLabelBlock.Input,
            output_schema=GithubAddLabelBlock.Output,
            test_input={
                "issue_url": "https://github.com/owner/repo/issues/1",
                "label": "bug",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Label added successfully")],
            test_mock={"add_label": lambda *args, **kwargs: "Label added successfully"},
        )

    @staticmethod
    def add_label(credentials: GithubCredentials, issue_url: str, label: str) -> str:
        try:
            # Convert the provided GitHub URL to the API URL
            if "/pull/" in issue_url:
                api_url = (
                    issue_url.replace("github.com", "api.github.com/repos").replace(
                        "/pull/", "/issues/"
                    )
                    + "/labels"
                )
            else:
                api_url = (
                    issue_url.replace("github.com", "api.github.com/repos") + "/labels"
                )

            # Log the constructed API URL for debugging
            print(f"Constructed API URL: {api_url}")

            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }
            data = {"labels": [label]}

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            return "Label added successfully"
        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error occurred: {http_err} - {http_err.response.text}"
        except Exception as e:
            return f"Failed to add label: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.add_label(
            credentials,
            input_data.issue_url,
            input_data.label,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status


class GithubRemoveLabelBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        issue_url: str = SchemaField(
            description="URL of the GitHub issue or pull request",
            placeholder="https://github.com/owner/repo/issues/1",
        )
        label: str = SchemaField(
            description="Label to remove from the issue or pull request",
            placeholder="Enter the label",
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Status of the label removal operation")
        error: str = SchemaField(
            description="Error message if the label removal failed"
        )

    def __init__(self):
        super().__init__(
            id="0012m3n4-5678-90ab-1234-567890abcdef",
            description="This block removes a label from a specified GitHub issue or pull request using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubRemoveLabelBlock.Input,
            output_schema=GithubRemoveLabelBlock.Output,
            test_input={
                "issue_url": "https://github.com/owner/repo/issues/1",
                "label": "bug",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Label removed successfully")],
            test_mock={
                "remove_label": lambda *args, **kwargs: "Label removed successfully"
            },
        )

    @staticmethod
    def remove_label(credentials: GithubCredentials, issue_url: str, label: str) -> str:
        try:
            # Convert the provided GitHub URL to the API URL
            if "/pull/" in issue_url:
                api_url = (
                    issue_url.replace("github.com", "api.github.com/repos").replace(
                        "/pull/", "/issues/"
                    )
                    + f"/labels/{label}"
                )
            else:
                api_url = (
                    issue_url.replace("github.com", "api.github.com/repos")
                    + f"/labels/{label}"
                )

            # Log the constructed API URL for debugging
            print(f"Constructed API URL: {api_url}")

            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.delete(api_url, headers=headers)
            response.raise_for_status()

            return "Label removed successfully"
        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error occurred: {http_err} - {http_err.response.text}"
        except Exception as e:
            return f"Failed to remove label: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.remove_label(
            credentials,
            input_data.issue_url,
            input_data.label,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status


class GithubAssignIssueBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        issue_url: str = SchemaField(
            description="URL of the GitHub issue",
            placeholder="https://github.com/owner/repo/issues/1",
        )
        assignee: str = SchemaField(
            description="Username to assign to the issue",
            placeholder="Enter the username",
        )

    class Output(BlockSchema):
        status: str = SchemaField(
            description="Status of the issue assignment operation"
        )
        error: str = SchemaField(
            description="Error message if the issue assignment failed"
        )

    def __init__(self):
        super().__init__(
            id="0004r3s5-6789-01bc-2345-678901bcdefg",
            description="This block assigns a user to a specified GitHub issue using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubAssignIssueBlock.Input,
            output_schema=GithubAssignIssueBlock.Output,
            test_input={
                "issue_url": "https://github.com/owner/repo/issues/1",
                "assignee": "username1",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Issue assigned successfully")],
            test_mock={
                "assign_issue": lambda *args, **kwargs: "Issue assigned successfully"
            },
        )

    @staticmethod
    def assign_issue(
        credentials: GithubCredentials,
        issue_url: str,
        assignee: str,
    ) -> str:
        try:
            # Extracting repo path and issue number from the issue URL
            repo_path, issue_number = issue_url.replace(
                "https://github.com/", ""
            ).split("/issues/")
            api_url = f"https://api.github.com/repos/{repo_path}/issues/{issue_number}/assignees"

            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }
            data = {"assignees": [assignee]}

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            return "Issue assigned successfully"
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to assign issue: {str(http_err)}"
        except Exception as e:
            return f"Failed to assign issue: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.assign_issue(
            credentials,
            input_data.issue_url,
            input_data.assignee,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status


class GithubUnassignIssueBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        issue_url: str = SchemaField(
            description="URL of the GitHub issue",
            placeholder="https://github.com/owner/repo/issues/1",
        )
        assignee: str = SchemaField(
            description="Username to unassign from the issue",
            placeholder="Enter the username",
        )

    class Output(BlockSchema):
        status: str = SchemaField(
            description="Status of the issue unassignment operation"
        )
        error: str = SchemaField(
            description="Error message if the issue unassignment failed"
        )

    def __init__(self):
        super().__init__(
            id="0005r3s6-7890-12cd-3456-789012cdefgh",
            description="This block unassigns a user from a specified GitHub issue using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubUnassignIssueBlock.Input,
            output_schema=GithubUnassignIssueBlock.Output,
            test_input={
                "issue_url": "https://github.com/owner/repo/issues/1",
                "assignee": "username1",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Issue unassigned successfully")],
            test_mock={
                "unassign_issue": lambda *args, **kwargs: "Issue unassigned successfully"
            },
        )

    @staticmethod
    def unassign_issue(
        credentials: GithubCredentials,
        issue_url: str,
        assignee: str,
    ) -> str:
        try:
            # Extracting repo path and issue number from the issue URL
            repo_path, issue_number = issue_url.replace(
                "https://github.com/", ""
            ).split("/issues/")
            api_url = f"https://api.github.com/repos/{repo_path}/issues/{issue_number}/assignees"

            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }
            data = {"assignees": [assignee]}

            response = requests.delete(api_url, headers=headers, json=data)
            response.raise_for_status()

            return "Issue unassigned successfully"
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to unassign issue: {str(http_err)}"
        except Exception as e:
            return f"Failed to unassign issue: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.unassign_issue(
            credentials,
            input_data.issue_url,
            input_data.assignee,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status
