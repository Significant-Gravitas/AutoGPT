import base64
from typing import Literal

import requests
from autogpt_libs.supabase_integration_credentials_store.types import (
    APIKeyCredentials,
    OAuth2Credentials,
)
from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsField, CredentialsMetaInput, SchemaField

GithubCredentials = APIKeyCredentials | OAuth2Credentials
GithubCredentialsInput = CredentialsMetaInput[
    Literal["github"], Literal["api_key", "oauth2"]
]


def GithubCredentialsField(scope: str) -> GithubCredentialsInput:
    """
    Creates a GitHub credentials input on a block.

    Params:
        scope: The authorization scope needed for the block to work. ([list of available scopes](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps#available-scopes))
    """  # noqa
    return CredentialsField(
        provider="github",
        supported_credential_types={"api_key", "oauth2"},
        required_scopes={scope},
        description="The GitHub integration can be used with OAuth, "
        "or any API key with sufficient permissions for the blocks it is used on.",
    )


TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="github",
    api_key=SecretStr("mock-github-api-key"),
    title="Mock GitHub API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


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


class GithubMakePRBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        title: str = SchemaField(
            description="Title of the pull request",
            placeholder="Enter the pull request title",
        )
        body: str = SchemaField(
            description="Body of the pull request",
            placeholder="Enter the pull request body",
        )
        head: str = SchemaField(
            description="The name of the branch where your changes are implemented. For cross-repository pull requests in the same network, namespace head with a user like this: username:branch.",
            placeholder="Enter the head branch",
        )
        base: str = SchemaField(
            description="The name of the branch you want the changes pulled into.",
            placeholder="Enter the base branch",
        )

    class Output(BlockSchema):
        status: str = SchemaField(
            description="Status of the pull request creation operation"
        )
        error: str = SchemaField(
            description="Error message if the pull request creation failed"
        )

    def __init__(self):
        super().__init__(
            id="0003q3r4-5678-90ab-1234-567890abcdef",
            description="This block creates a new pull request on a specified GitHub repository using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubMakePRBlock.Input,
            output_schema=GithubMakePRBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "title": "Test Pull Request",
                "body": "This is a test pull request.",
                "head": "feature-branch",
                "base": "main",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Pull request created successfully")],
            test_mock={
                "create_pr": lambda *args, **kwargs: "Pull request created successfully"
            },
        )

    @staticmethod
    def create_pr(
        credentials: GithubCredentials,
        repo_url: str,
        title: str,
        body: str,
        head: str,
        base: str,
    ) -> str:
        response = None
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = f"https://api.github.com/repos/{repo_path}/pulls"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }
            data = {"title": title, "body": body, "head": head, "base": base}

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            return "Pull request created successfully"
        except requests.exceptions.HTTPError as http_err:
            if response and response.status_code == 422:
                error_details = response.json()
                return f"Failed to create pull request: {error_details.get('message', 'Unknown error')}"
            return f"Failed to create pull request: {str(http_err)}"
        except Exception as e:
            return f"Failed to create pull request: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.create_pr(
            credentials,
            input_data.repo_url,
            input_data.title,
            input_data.body,
            input_data.head,
            input_data.base,
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


class GithubReadPRBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
        )
        include_pr_changes: bool = SchemaField(
            description="Whether to include the changes made in the pull request",
            default=False,
        )

    class Output(BlockSchema):
        title: str = SchemaField(description="Title of the pull request")
        body: str = SchemaField(description="Body of the pull request")
        user: str = SchemaField(description="User who created the pull request")
        changes: str = SchemaField(description="Changes made in the pull request")
        error: str = SchemaField(
            description="Error message if reading the pull request failed"
        )

    def __init__(self):
        super().__init__(
            id="0005g3h4-5678-90ab-1234-567890abcdeg",
            description="This block reads the body, title, user, and changes of a specified GitHub pull request using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadPRBlock.Input,
            output_schema=GithubReadPRBlock.Output,
            test_input={
                "pr_url": "https://github.com/owner/repo/pull/1",
                "include_pr_changes": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("title", "Title of the pull request"),
                ("body", "This is the body of the pull request."),
                ("user", "username"),
                ("changes", "List of changes made in the pull request."),
            ],
            test_mock={
                "read_pr": lambda *args, **kwargs: (
                    "Title of the pull request",
                    "This is the body of the pull request.",
                    "username",
                ),
                "read_pr_changes": lambda *args, **kwargs: "List of changes made in the pull request.",
            },
        )

    @staticmethod
    def read_pr(credentials: GithubCredentials, pr_url: str) -> tuple[str, str, str]:
        try:
            api_url = pr_url.replace("github.com", "api.github.com/repos").replace(
                "/pull/", "/issues/"
            )

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
            return f"Failed to read pull request: {str(e)}", "", ""

    @staticmethod
    def read_pr_changes(credentials: GithubCredentials, pr_url: str) -> str:
        try:
            api_url = (
                pr_url.replace("github.com", "api.github.com/repos").replace(
                    "/pull/", "/pulls/"
                )
                + "/files"
            )

            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            files = response.json()
            changes = []
            for file in files:
                filename = file.get("filename")
                patch = file.get("patch")
                if filename and patch:
                    changes.append(f"File: {filename}\n{patch}")

            return "\n\n".join(changes)
        except Exception as e:
            return f"Failed to read PR changes: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        title, body, user = self.read_pr(
            credentials,
            input_data.pr_url,
        )
        if "Failed" in title:
            yield "error", title
        else:
            yield "title", title
            yield "body", body
            yield "user", user

        if input_data.include_pr_changes:
            changes = self.read_pr_changes(
                credentials,
                input_data.pr_url,
            )
            if "Failed" in changes:
                yield "error", changes
            else:
                yield "changes", changes
        else:
            yield "changes", "Changes not included"


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


class GithubReadTagsBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        tags: list[dict[str, str]] = SchemaField(
            description="List of tags with their names and URLs"
        )
        error: str = SchemaField(description="Error message if listing tags failed")

    def __init__(self):
        super().__init__(
            id="0007g3h4-5678-90ab-1234-567890abcdef",
            description="This block lists all tags for a specified GitHub repository using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadTagsBlock.Input,
            output_schema=GithubReadTagsBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "tags",
                    [
                        {
                            "name": "v1.0.0",
                            "url": "https://github.com/owner/repo/tree/v1.0.0",
                        }
                    ],
                )
            ],
            test_mock={
                "list_tags": lambda *args, **kwargs: [
                    {
                        "name": "v1.0.0",
                        "url": "https://github.com/owner/repo/tree/v1.0.0",
                    }
                ]
            },
        )

    @staticmethod
    def list_tags(
        credentials: GithubCredentials, repo_url: str
    ) -> list[dict[str, str]]:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = f"https://api.github.com/repos/{repo_path}/tags"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            tags = [
                {
                    "name": tag["name"],
                    "url": f"https://github.com/{repo_path}/tree/{tag['name']}",
                }
                for tag in data
            ]

            return tags
        except Exception as e:
            return [{"name": "Error", "url": f"Failed to list tags: {str(e)}"}]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        tags = self.list_tags(
            credentials,
            input_data.repo_url,
        )
        if any("Failed" in tag["url"] for tag in tags):
            yield "error", tags[0]["url"]
        else:
            yield "tags", tags


class GithubReadBranchesBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        branches: list[dict[str, str]] = SchemaField(
            description="List of branches with their names and URLs"
        )
        error: str = SchemaField(description="Error message if listing branches failed")

    def __init__(self):
        super().__init__(
            id="0008i3j4-5678-90ab-1234-567890abcdef",
            description="This block lists all branches for a specified GitHub repository using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadBranchesBlock.Input,
            output_schema=GithubReadBranchesBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "branches",
                    [
                        {
                            "name": "main",
                            "url": "https://github.com/owner/repo/tree/main",
                        }
                    ],
                )
            ],
            test_mock={
                "list_branches": lambda *args, **kwargs: [
                    {
                        "name": "main",
                        "url": "https://github.com/owner/repo/tree/main",
                    }
                ]
            },
        )

    @staticmethod
    def list_branches(
        credentials: GithubCredentials, repo_url: str
    ) -> list[dict[str, str]]:
        try:
            api_url = (
                repo_url.replace("github.com", "api.github.com/repos") + "/branches"
            )
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            branches = [
                {"name": branch["name"], "url": branch["commit"]["url"]}
                for branch in data
            ]

            return branches
        except Exception as e:
            return [{"name": "Error", "url": f"Failed to list branches: {str(e)}"}]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        branches = self.list_branches(
            credentials,
            input_data.repo_url,
        )
        if any("Failed" in branch["url"] for branch in branches):
            yield "error", branches[0]["url"]
        else:
            yield "branches", branches


class GithubReadDiscussionsBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        num_discussions: int = SchemaField(
            description="Number of discussions to fetch", default=5
        )

    class Output(BlockSchema):
        discussions: list[dict[str, str]] = SchemaField(
            description="List of discussions with their titles and URLs"
        )
        error: str = SchemaField(
            description="Error message if listing discussions failed"
        )

    def __init__(self):
        super().__init__(
            id="0009j3k4-5678-90ab-1234-567890abcdef",
            description="This block lists recent discussions for a specified GitHub repository using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadDiscussionsBlock.Input,
            output_schema=GithubReadDiscussionsBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "num_discussions": 3,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "discussions",
                    [
                        {
                            "title": "Discussion 1",
                            "url": "https://github.com/owner/repo/discussions/1",
                        }
                    ],
                )
            ],
            test_mock={
                "list_discussions": lambda *args, **kwargs: [
                    {
                        "title": "Discussion 1",
                        "url": "https://github.com/owner/repo/discussions/1",
                    }
                ]
            },
        )

    @staticmethod
    def list_discussions(
        credentials: GithubCredentials, repo_url: str, num_discussions: int
    ) -> list[dict[str, str]]:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            owner, repo = repo_path.split("/")
            query = """
            query($owner: String!, $repo: String!, $num: Int!) {
                repository(owner: $owner, name: $repo) {
                    discussions(first: $num) {
                        nodes {
                            title
                            url
                        }
                    }
                }
            }
            """
            variables = {"owner": owner, "repo": repo, "num": num_discussions}
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.post(
                "https://api.github.com/graphql",
                json={"query": query, "variables": variables},
                headers=headers,
            )
            response.raise_for_status()

            data = response.json()
            discussions = [
                {"title": discussion["title"], "url": discussion["url"]}
                for discussion in data["data"]["repository"]["discussions"]["nodes"]
            ]

            return discussions
        except Exception as e:
            return [{"title": "Error", "url": f"Failed to list discussions: {str(e)}"}]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        discussions = self.list_discussions(
            credentials, input_data.repo_url, input_data.num_discussions
        )
        if any("Failed" in discussion["url"] for discussion in discussions):
            yield "error", discussions[0]["url"]
        else:
            yield "discussions", discussions


class GithubReadReleasesBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        releases: list[dict[str, str]] = SchemaField(
            description="List of releases with their names and URLs"
        )
        error: str = SchemaField(description="Error message if listing releases failed")

    def __init__(self):
        super().__init__(
            id="0010k3l4-5678-90ab-1234-567890abcdef",
            description="This block lists all releases for a specified GitHub repository using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadReleasesBlock.Input,
            output_schema=GithubReadReleasesBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "releases",
                    [
                        {
                            "name": "v1.0.0",
                            "url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                        }
                    ],
                )
            ],
            test_mock={
                "list_releases": lambda *args, **kwargs: [
                    {
                        "name": "v1.0.0",
                        "url": "https://github.com/owner/repo/releases/tag/v1.0.0",
                    }
                ]
            },
        )

    @staticmethod
    def list_releases(
        credentials: GithubCredentials, repo_url: str
    ) -> list[dict[str, str]]:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = f"https://api.github.com/repos/{repo_path}/releases"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            releases = [
                {"name": release["name"], "url": release["html_url"]}
                for release in data
            ]

            return releases
        except Exception as e:
            return [{"name": "Error", "url": f"Failed to list releases: {str(e)}"}]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        releases = self.list_releases(
            credentials,
            input_data.repo_url,
        )
        if any("Failed" in release["url"] for release in releases):
            yield "error", releases[0]["url"]
        else:
            yield "releases", releases


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


class GithubAssignReviewerBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
        )
        reviewer: str = SchemaField(
            description="Username of the reviewer to assign",
            placeholder="Enter the reviewer's username",
        )

    class Output(BlockSchema):
        status: str = SchemaField(
            description="Status of the reviewer assignment operation"
        )
        error: str = SchemaField(
            description="Error message if the reviewer assignment failed"
        )

    def __init__(self):
        super().__init__(
            id="0014o3p4-5678-90ab-1234-567890abcdef",
            description="This block assigns a reviewer to a specified GitHub pull request using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubAssignReviewerBlock.Input,
            output_schema=GithubAssignReviewerBlock.Output,
            test_input={
                "pr_url": "https://github.com/owner/repo/pull/1",
                "reviewer": "reviewer_username",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Reviewer assigned successfully")],
            test_mock={
                "assign_reviewer": lambda *args, **kwargs: "Reviewer assigned successfully"
            },
        )

    @staticmethod
    def assign_reviewer(
        credentials: GithubCredentials, pr_url: str, reviewer: str
    ) -> str:
        try:
            # Convert the PR URL to the appropriate API endpoint
            api_url = (
                pr_url.replace("github.com", "api.github.com/repos").replace(
                    "/pull/", "/pulls/"
                )
                + "/requested_reviewers"
            )

            # Log the constructed API URL for debugging
            print(f"Constructed API URL: {api_url}")

            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }
            data = {"reviewers": [reviewer]}

            # Log the request data for debugging
            print(f"Request data: {data}")

            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()

            return "Reviewer assigned successfully"
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 422:
                return f"Failed to assign reviewer: The reviewer '{reviewer}' may not have permission or the pull request is not in a valid state. Detailed error: {http_err.response.text}"
            else:
                return f"HTTP error occurred: {http_err} - {http_err.response.text}"
        except Exception as e:
            return f"Failed to assign reviewer: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.assign_reviewer(
            credentials,
            input_data.pr_url,
            input_data.reviewer,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status


class GithubUnassignReviewerBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
        )
        reviewer: str = SchemaField(
            description="Username of the reviewer to unassign",
            placeholder="Enter the reviewer's username",
        )

    class Output(BlockSchema):
        status: str = SchemaField(
            description="Status of the reviewer unassignment operation"
        )
        error: str = SchemaField(
            description="Error message if the reviewer unassignment failed"
        )

    def __init__(self):
        super().__init__(
            id="0015p3q4-5678-90ab-1234-567890abcdef",
            description="This block unassigns a reviewer from a specified GitHub pull request using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubUnassignReviewerBlock.Input,
            output_schema=GithubUnassignReviewerBlock.Output,
            test_input={
                "pr_url": "https://github.com/owner/repo/pull/1",
                "reviewer": "reviewer_username",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Reviewer unassigned successfully")],
            test_mock={
                "unassign_reviewer": lambda *args, **kwargs: "Reviewer unassigned successfully"
            },
        )

    @staticmethod
    def unassign_reviewer(
        credentials: GithubCredentials, pr_url: str, reviewer: str
    ) -> str:
        try:
            api_url = (
                pr_url.replace("github.com", "api.github.com/repos").replace(
                    "/pull/", "/pulls/"
                )
                + "/requested_reviewers"
            )
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }
            data = {"reviewers": [reviewer]}

            response = requests.delete(api_url, headers=headers, json=data)
            response.raise_for_status()

            return "Reviewer unassigned successfully"
        except Exception as e:
            return f"Failed to unassign reviewer: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.unassign_reviewer(
            credentials,
            input_data.pr_url,
            input_data.reviewer,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status


class GithubListReviewersBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
        )

    class Output(BlockSchema):
        reviewers: list[dict[str, str]] = SchemaField(
            description="List of reviewers with their usernames and URLs"
        )
        error: str = SchemaField(
            description="Error message if listing reviewers failed"
        )

    def __init__(self):
        super().__init__(
            id="0016q3r4-5678-90ab-1234-567890abcdef",
            description="This block lists all reviewers for a specified GitHub pull request using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListReviewersBlock.Input,
            output_schema=GithubListReviewersBlock.Output,
            test_input={
                "pr_url": "https://github.com/owner/repo/pull/1",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "reviewers",
                    [
                        {
                            "username": "reviewer1",
                            "url": "https://github.com/reviewer1",
                        }
                    ],
                )
            ],
            test_mock={
                "list_reviewers": lambda *args, **kwargs: [
                    {
                        "username": "reviewer1",
                        "url": "https://github.com/reviewer1",
                    }
                ]
            },
        )

    @staticmethod
    def list_reviewers(
        credentials: GithubCredentials, pr_url: str
    ) -> list[dict[str, str]]:
        try:
            api_url = (
                pr_url.replace("github.com", "api.github.com/repos").replace(
                    "/pull/", "/pulls/"
                )
                + "/requested_reviewers"
            )
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            reviewers = [
                {"username": reviewer["login"], "url": reviewer["html_url"]}
                for reviewer in data.get("users", [])
            ]

            return reviewers
        except Exception as e:
            return [{"username": "Error", "url": f"Failed to list reviewers: {str(e)}"}]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        reviewers = self.list_reviewers(
            credentials,
            input_data.pr_url,
        )
        if any("Failed" in reviewer["url"] for reviewer in reviewers):
            yield "error", reviewers[0]["url"]
        else:
            yield "reviewers", reviewers


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


class GithubReadCodeownersFileBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        codeowners_content: str = SchemaField(
            description="Content of the CODEOWNERS file"
        )
        error: str = SchemaField(description="Error message if the file reading failed")

    def __init__(self):
        super().__init__(
            id="0006r3s7-8901-23de-4567-890123defghi",
            description="This block reads the CODEOWNERS file from the master branch of a specified GitHub repository using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadCodeownersFileBlock.Input,
            output_schema=GithubReadCodeownersFileBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("codeowners_content", "# CODEOWNERS content")],
            test_mock={
                "read_codeowners": lambda *args, **kwargs: "# CODEOWNERS content"
            },
        )

    @staticmethod
    def read_codeowners(credentials: GithubCredentials, repo_url: str) -> str:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = f"https://api.github.com/repos/{repo_path}/contents/.github/CODEOWNERS?ref=master"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            content = response.json()
            return base64.b64decode(content["content"]).decode("utf-8")
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to read CODEOWNERS file: {str(http_err)}"
        except Exception as e:
            return f"Failed to read CODEOWNERS file: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        content = self.read_codeowners(
            credentials,
            input_data.repo_url,
        )
        if "Failed" not in content:
            yield "codeowners_content", content
        else:
            yield "error", content


class GithubReadFileFromMasterBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        file_path: str = SchemaField(
            description="Path to the file in the repository",
            placeholder="path/to/file",
        )

    class Output(BlockSchema):
        file_content: str = SchemaField(
            description="Content of the file from the master branch"
        )
        error: str = SchemaField(description="Error message if the file reading failed")

    def __init__(self):
        super().__init__(
            id="0007r3s8-9012-34ef-5678-901234efghij",
            description="This block reads the content of a specified file from the master branch of a GitHub repository using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadFileFromMasterBlock.Input,
            output_schema=GithubReadFileFromMasterBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "file_path": "path/to/file",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("file_content", "File content")],
            test_mock={"read_file": lambda *args, **kwargs: "File content"},
        )

    @staticmethod
    def read_file(credentials: GithubCredentials, repo_url: str, file_path: str) -> str:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = f"https://api.github.com/repos/{repo_path}/contents/{file_path}?ref=master"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            content = response.json()
            return base64.b64decode(content["content"]).decode("utf-8")
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to read file: {str(http_err)}"
        except Exception as e:
            return f"Failed to read file: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        content = self.read_file(
            credentials,
            input_data.repo_url,
            input_data.file_path,
        )
        if "Failed" not in content:
            yield "file_content", content
        else:
            yield "error", content


class GithubReadFileFolderRepoBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        path: str = SchemaField(
            description="Path to the file/folder in the repository",
            placeholder="path/to/file_or_folder",
        )
        branch: str = SchemaField(
            description="Branch name to read from",
            placeholder="branch_name",
        )

    class Output(BlockSchema):
        content: str = SchemaField(
            description="Content of the file/folder/repo from the specified branch"
        )
        error: str = SchemaField(description="Error message if the reading failed")

    def __init__(self):
        super().__init__(
            id="0008r3s9-0123-45fg-6789-012345fghijk",
            description="This block reads the content of a specified file, folder, or repository from a specified branch using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadFileFolderRepoBlock.Input,
            output_schema=GithubReadFileFolderRepoBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "path": "path/to/file_or_folder",
                "branch": "branch_name",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("content", "File or folder content")],
            test_mock={
                "read_content": lambda *args, **kwargs: "File or folder content"
            },
        )

    @staticmethod
    def read_content(
        credentials: GithubCredentials, repo_url: str, path: str, branch: str
    ) -> str:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = (
                f"https://api.github.com/repos/{repo_path}/contents/{path}?ref={branch}"
            )
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            content = response.json()
            if "content" in content:
                return base64.b64decode(content["content"]).decode("utf-8")
            else:
                return content  # Return the directory content as JSON

        except requests.exceptions.HTTPError as http_err:
            return f"Failed to read content: {str(http_err)}"
        except Exception as e:
            return f"Failed to read content: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        content = self.read_content(
            credentials,
            input_data.repo_url,
            input_data.path,
            input_data.branch,
        )
        if "Failed" not in content:
            yield "content", content
        else:
            yield "error", content


class GithubMakeBranchBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        new_branch: str = SchemaField(
            description="Name of the new branch",
            placeholder="new_branch_name",
        )
        source_branch: str = SchemaField(
            description="Name of the source branch",
            placeholder="source_branch_name",
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Status of the branch creation operation")
        error: str = SchemaField(
            description="Error message if the branch creation failed"
        )

    def __init__(self):
        super().__init__(
            id="0008r3s9-0123-45fg-6789-012345fghijp",
            description="This block creates a new branch from a specified source branch using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubMakeBranchBlock.Input,
            output_schema=GithubMakeBranchBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "new_branch": "new_branch_name",
                "source_branch": "source_branch_name",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Branch created successfully")],
            test_mock={
                "create_branch": lambda *args, **kwargs: "Branch created successfully"
            },
        )

    @staticmethod
    def create_branch(
        credentials: GithubCredentials,
        repo_url: str,
        new_branch: str,
        source_branch: str,
    ) -> str:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            ref_api_url = f"https://api.github.com/repos/{repo_path}/git/refs/heads/{source_branch}"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(ref_api_url, headers=headers)
            response.raise_for_status()

            sha = response.json()["object"]["sha"]

            create_branch_api_url = f"https://api.github.com/repos/{repo_path}/git/refs"
            data = {"ref": f"refs/heads/{new_branch}", "sha": sha}

            response = requests.post(create_branch_api_url, headers=headers, json=data)
            response.raise_for_status()

            return "Branch created successfully"
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to create branch: {str(http_err)}"
        except Exception as e:
            return f"Failed to create branch: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.create_branch(
            credentials,
            input_data.repo_url,
            input_data.new_branch,
            input_data.source_branch,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status


class GithubDeleteBranchBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        branch: str = SchemaField(
            description="Name of the branch to delete",
            placeholder="branch_name",
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Status of the branch deletion operation")
        error: str = SchemaField(
            description="Error message if the branch deletion failed"
        )

    def __init__(self):
        super().__init__(
            id="0008r3s9-0123-45fg-6789-012345fghijq",
            description="This block deletes a specified branch using OAuth credentials.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubDeleteBranchBlock.Input,
            output_schema=GithubDeleteBranchBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "branch": "branch_name",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Branch deleted successfully")],
            test_mock={
                "delete_branch": lambda *args, **kwargs: "Branch deleted successfully"
            },
        )

    @staticmethod
    def delete_branch(
        credentials: GithubCredentials, repo_url: str, branch: str
    ) -> str:
        try:
            repo_path = repo_url.replace("https://github.com/", "")
            api_url = (
                f"https://api.github.com/repos/{repo_path}/git/refs/heads/{branch}"
            )
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.delete(api_url, headers=headers)
            response.raise_for_status()

            return "Branch deleted successfully"
        except requests.exceptions.HTTPError as http_err:
            return f"Failed to delete branch: {str(http_err)}"
        except Exception as e:
            return f"Failed to delete branch: {str(e)}"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        status = self.delete_branch(
            credentials,
            input_data.repo_url,
            input_data.branch,
        )
        if "successfully" in status:
            yield "status", status
        else:
            yield "error", status
