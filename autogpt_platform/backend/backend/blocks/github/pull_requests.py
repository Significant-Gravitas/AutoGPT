import base64
import re

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


class GithubListPullRequestsBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        class PRItem(TypedDict):
            title: str
            url: str

        pull_request: PRItem = SchemaField(
            title="Pull Request", description="PRs with their title and URL"
        )
        error: str = SchemaField(description="Error message if listing issues failed")

    def __init__(self):
        super().__init__(
            id="ffef3c4c-6cd0-48dd-817d-459f975219f4",
            description="This block lists all pull requests for a specified GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListPullRequestsBlock.Input,
            output_schema=GithubListPullRequestsBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "pull_request",
                    {
                        "title": "Pull request 1",
                        "url": "https://github.com/owner/repo/pull/1",
                    },
                )
            ],
            test_mock={
                "list_prs": lambda *args, **kwargs: [
                    {
                        "title": "Pull request 1",
                        "url": "https://github.com/owner/repo/pull/1",
                    }
                ]
            },
        )

    @staticmethod
    def list_prs(credentials: GithubCredentials, repo_url: str) -> list[Output.PRItem]:
        api = get_api(credentials)
        pulls_url = repo_url + "/pulls"
        response = api.get(pulls_url)
        data = response.json()
        pull_requests: list[GithubListPullRequestsBlock.Output.PRItem] = [
            {"title": pr["title"], "url": pr["html_url"]} for pr in data
        ]
        return pull_requests

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        pull_requests = self.list_prs(
            credentials,
            input_data.repo_url,
        )
        yield from (("pull_request", pr) for pr in pull_requests)


class GithubMakePullRequestBlock(Block):
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
            description=(
                "The name of the branch where your changes are implemented. "
                "For cross-repository pull requests in the same network, "
                "namespace head with a user like this: username:branch."
            ),
            placeholder="Enter the head branch",
        )
        base: str = SchemaField(
            description="The name of the branch you want the changes pulled into.",
            placeholder="Enter the base branch",
        )

    class Output(BlockSchema):
        number: int = SchemaField(description="Number of the created pull request")
        url: str = SchemaField(description="URL of the created pull request")
        error: str = SchemaField(
            description="Error message if the pull request creation failed"
        )

    def __init__(self):
        super().__init__(
            id="dfb987f8-f197-4b2e-bf19-111812afd692",
            description="This block creates a new pull request on a specified GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubMakePullRequestBlock.Input,
            output_schema=GithubMakePullRequestBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "title": "Test Pull Request",
                "body": "This is a test pull request.",
                "head": "feature-branch",
                "base": "main",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("number", 1),
                ("url", "https://github.com/owner/repo/pull/1"),
            ],
            test_mock={
                "create_pr": lambda *args, **kwargs: (
                    1,
                    "https://github.com/owner/repo/pull/1",
                )
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
    ) -> tuple[int, str]:
        api = get_api(credentials)
        pulls_url = repo_url + "/pulls"
        data = {"title": title, "body": body, "head": head, "base": base}
        response = api.post(pulls_url, json=data)
        pr_data = response.json()
        return pr_data["number"], pr_data["html_url"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            number, url = self.create_pr(
                credentials,
                input_data.repo_url,
                input_data.title,
                input_data.body,
                input_data.head,
                input_data.base,
            )
            yield "number", number
            yield "url", url
        except Exception as e:
            yield "error", str(e)


class GithubReadPullRequestBlock(Block):
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
        author: str = SchemaField(description="User who created the pull request")
        changes: str = SchemaField(description="Changes made in the pull request")
        error: str = SchemaField(
            description="Error message if reading the pull request failed"
        )

    def __init__(self):
        super().__init__(
            id="bf94b2a4-1a30-4600-a783-a8a44ee31301",
            description="This block reads the body, title, user, and changes of a specified GitHub pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubReadPullRequestBlock.Input,
            output_schema=GithubReadPullRequestBlock.Output,
            test_input={
                "pr_url": "https://github.com/owner/repo/pull/1",
                "include_pr_changes": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("title", "Title of the pull request"),
                ("body", "This is the body of the pull request."),
                ("author", "username"),
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
        api = get_api(credentials)
        # Adjust the URL to access the issue endpoint for PR metadata
        issue_url = pr_url.replace("/pull/", "/issues/")
        response = api.get(issue_url)
        data = response.json()
        title = data.get("title", "No title found")
        body = data.get("body", "No body content found")
        author = data.get("user", {}).get("login", "No user found")
        return title, body, author

    @staticmethod
    def read_pr_changes(credentials: GithubCredentials, pr_url: str) -> str:
        api = get_api(credentials)
        files_url = prepare_pr_api_url(pr_url=pr_url, path="files")
        response = api.get(files_url)
        files = response.json()
        changes = []
        for file in files:
            filename = file.get("filename")
            patch = file.get("patch")
            if filename and patch:
                changes.append(f"File: {filename}\n{patch}")
        return "\n\n".join(changes)

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        title, body, author = self.read_pr(
            credentials,
            input_data.pr_url,
        )
        yield "title", title
        yield "body", body
        yield "author", author

        if input_data.include_pr_changes:
            changes = self.read_pr_changes(
                credentials,
                input_data.pr_url,
            )
            yield "changes", changes


class GithubAssignPRReviewerBlock(Block):
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
            id="c0d22c5e-e688-43e3-ba43-d5faba7927fd",
            description="This block assigns a reviewer to a specified GitHub pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubAssignPRReviewerBlock.Input,
            output_schema=GithubAssignPRReviewerBlock.Output,
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
        api = get_api(credentials)
        reviewers_url = prepare_pr_api_url(pr_url=pr_url, path="requested_reviewers")
        data = {"reviewers": [reviewer]}
        api.post(reviewers_url, json=data)
        return "Reviewer assigned successfully"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            status = self.assign_reviewer(
                credentials,
                input_data.pr_url,
                input_data.reviewer,
            )
            yield "status", status
        except Exception as e:
            yield "error", str(e)


class GithubUnassignPRReviewerBlock(Block):
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
            id="9637945d-c602-4875-899a-9c22f8fd30de",
            description="This block unassigns a reviewer from a specified GitHub pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubUnassignPRReviewerBlock.Input,
            output_schema=GithubUnassignPRReviewerBlock.Output,
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
        api = get_api(credentials)
        reviewers_url = prepare_pr_api_url(pr_url=pr_url, path="requested_reviewers")
        data = {"reviewers": [reviewer]}
        api.delete(reviewers_url, json=data)
        return "Reviewer unassigned successfully"

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            status = self.unassign_reviewer(
                credentials,
                input_data.pr_url,
                input_data.reviewer,
            )
            yield "status", status
        except Exception as e:
            yield "error", str(e)


class GithubListPRReviewersBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
        )

    class Output(BlockSchema):
        class ReviewerItem(TypedDict):
            username: str
            url: str

        reviewer: ReviewerItem = SchemaField(
            title="Reviewer",
            description="Reviewers with their username and profile URL",
        )
        error: str = SchemaField(
            description="Error message if listing reviewers failed"
        )

    def __init__(self):
        super().__init__(
            id="2646956e-96d5-4754-a3df-034017e7ed96",
            description="This block lists all reviewers for a specified GitHub pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListPRReviewersBlock.Input,
            output_schema=GithubListPRReviewersBlock.Output,
            test_input={
                "pr_url": "https://github.com/owner/repo/pull/1",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "reviewer",
                    {
                        "username": "reviewer1",
                        "url": "https://github.com/reviewer1",
                    },
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
    ) -> list[Output.ReviewerItem]:
        api = get_api(credentials)
        reviewers_url = prepare_pr_api_url(pr_url=pr_url, path="requested_reviewers")
        response = api.get(reviewers_url)
        data = response.json()
        reviewers: list[GithubListPRReviewersBlock.Output.ReviewerItem] = [
            {"username": reviewer["login"], "url": reviewer["html_url"]}
            for reviewer in data.get("users", [])
        ]
        return reviewers

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
        yield from (("reviewer", reviewer) for reviewer in reviewers)


def prepare_pr_api_url(pr_url: str, path: str) -> str:
    # Pattern to capture the base repository URL and the pull request number
    pattern = r"^(?:https?://)?([^/]+/[^/]+/[^/]+)/pull/(\d+)"
    match = re.match(pattern, pr_url)
    if not match:
        return pr_url

    base_url, pr_number = match.groups()
    return f"{base_url}/pulls/{pr_number}/{path}"


class GithubCreateFileBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        file_path: str = SchemaField(
            description="Path where the file should be created",
            placeholder="path/to/file.txt",
        )
        content: str = SchemaField(
            description="Content to write to the file",
            placeholder="File content here",
        )
        branch: str = SchemaField(
            description="Branch where the file should be created",
            default="main",
        )
        commit_message: str = SchemaField(
            description="Message for the commit",
            default="Create new file",
        )

    class Output(BlockSchema):
        url: str = SchemaField(description="URL of the created file")
        sha: str = SchemaField(description="SHA of the commit")
        error: str = SchemaField(
            description="Error message if the file creation failed"
        )

    def __init__(self):
        super().__init__(
            id="8fd132ac-b917-428a-8159-d62893e8a3fe",
            description="This block creates a new file in a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreateFileBlock.Input,
            output_schema=GithubCreateFileBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "file_path": "test/file.txt",
                "content": "Test content",
                "branch": "main",
                "commit_message": "Create test file",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("url", "https://github.com/owner/repo/blob/main/test/file.txt"),
                ("sha", "abc123"),
            ],
            test_mock={
                "create_file": lambda *args, **kwargs: (
                    "https://github.com/owner/repo/blob/main/test/file.txt",
                    "abc123",
                )
            },
        )

    @staticmethod
    def create_file(
        credentials: GithubCredentials,
        repo_url: str,
        file_path: str,
        content: str,
        branch: str,
        commit_message: str,
    ) -> tuple[str, str]:
        api = get_api(credentials)
        # Convert content to base64
        content_bytes = content.encode("utf-8")
        content_base64 = base64.b64encode(content_bytes).decode("utf-8")

        # Create the file using the GitHub API
        contents_url = f"{repo_url}/contents/{file_path}"
        data = {
            "message": commit_message,
            "content": content_base64,
            "branch": branch,
        }
        response = api.put(contents_url, json=data)
        result = response.json()

        return result["content"]["html_url"], result["commit"]["sha"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, sha = self.create_file(
                credentials,
                input_data.repo_url,
                input_data.file_path,
                input_data.content,
                input_data.branch,
                input_data.commit_message,
            )
            yield "url", url
            yield "sha", sha
        except Exception as e:
            yield "error", str(e)


class GithubUpdateFileBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )
        file_path: str = SchemaField(
            description="Path to the file to update",
            placeholder="path/to/file.txt",
        )
        content: str = SchemaField(
            description="New content for the file",
            placeholder="Updated content here",
        )
        branch: str = SchemaField(
            description="Branch containing the file",
            default="main",
        )
        commit_message: str = SchemaField(
            description="Message for the commit",
            default="Update file",
        )

    class Output(BlockSchema):
        url: str = SchemaField(description="URL of the updated file")
        sha: str = SchemaField(description="SHA of the commit")
        error: str = SchemaField(description="Error message if the file update failed")

    def __init__(self):
        super().__init__(
            id="30be12a4-57cb-4aa4-baf5-fcc68d136076",
            description="This block updates an existing file in a GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubUpdateFileBlock.Input,
            output_schema=GithubUpdateFileBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "file_path": "test/file.txt",
                "content": "Updated content",
                "branch": "main",
                "commit_message": "Update test file",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("url", "https://github.com/owner/repo/blob/main/test/file.txt"),
                ("sha", "def456"),
            ],
            test_mock={
                "update_file": lambda *args, **kwargs: (
                    "https://github.com/owner/repo/blob/main/test/file.txt",
                    "def456",
                )
            },
        )

    @staticmethod
    def update_file(
        credentials: GithubCredentials,
        repo_url: str,
        file_path: str,
        content: str,
        branch: str,
        commit_message: str,
    ) -> tuple[str, str]:
        api = get_api(credentials)

        # First get the current file to get its SHA
        contents_url = f"{repo_url}/contents/{file_path}"
        params = {"ref": branch}
        response = api.get(contents_url, params=params)
        current_file = response.json()

        # Convert new content to base64
        content_bytes = content.encode("utf-8")
        content_base64 = base64.b64encode(content_bytes).decode("utf-8")

        # Update the file
        data = {
            "message": commit_message,
            "content": content_base64,
            "sha": current_file["sha"],
            "branch": branch,
        }
        response = api.put(contents_url, json=data)
        result = response.json()

        return result["content"]["html_url"], result["commit"]["sha"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, sha = self.update_file(
                credentials,
                input_data.repo_url,
                input_data.file_path,
                input_data.content,
                input_data.branch,
                input_data.commit_message,
            )
            yield "url", url
            yield "sha", sha
        except Exception as e:
            yield "error", str(e)


class GithubCreateRepositoryBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        name: str = SchemaField(
            description="Name of the repository to create",
            placeholder="my-new-repo",
        )
        description: str = SchemaField(
            description="Description of the repository",
            placeholder="A description of the repository",
            default="",
        )
        private: bool = SchemaField(
            description="Whether the repository should be private",
            default=False,
        )
        auto_init: bool = SchemaField(
            description="Whether to initialize the repository with a README",
            default=True,
        )
        gitignore_template: str = SchemaField(
            description="Git ignore template to use (e.g., Python, Node, Java)",
            default="",
        )

    class Output(BlockSchema):
        url: str = SchemaField(description="URL of the created repository")
        clone_url: str = SchemaField(description="Git clone URL of the repository")
        error: str = SchemaField(
            description="Error message if the repository creation failed"
        )

    def __init__(self):
        super().__init__(
            id="029ec3b8-1cfd-46d3-b6aa-28e4a706efd1",
            description="This block creates a new GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreateRepositoryBlock.Input,
            output_schema=GithubCreateRepositoryBlock.Output,
            test_input={
                "name": "test-repo",
                "description": "A test repository",
                "private": False,
                "auto_init": True,
                "gitignore_template": "Python",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("url", "https://github.com/owner/test-repo"),
                ("clone_url", "https://github.com/owner/test-repo.git"),
            ],
            test_mock={
                "create_repository": lambda *args, **kwargs: (
                    "https://github.com/owner/test-repo",
                    "https://github.com/owner/test-repo.git",
                )
            },
        )

    @staticmethod
    def create_repository(
        credentials: GithubCredentials,
        name: str,
        description: str,
        private: bool,
        auto_init: bool,
        gitignore_template: str,
    ) -> tuple[str, str]:
        api = get_api(credentials, convert_urls=False)  # Disable URL conversion
        data = {
            "name": name,
            "description": description,
            "private": private,
            "auto_init": auto_init,
        }

        if gitignore_template:
            data["gitignore_template"] = gitignore_template

        # Create repository using the user endpoint
        response = api.post("https://api.github.com/user/repos", json=data)
        result = response.json()

        return result["html_url"], result["clone_url"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            url, clone_url = self.create_repository(
                credentials,
                input_data.name,
                input_data.description,
                input_data.private,
                input_data.auto_init,
                input_data.gitignore_template,
            )
            yield "url", url
            yield "clone_url", clone_url
        except Exception as e:
            yield "error", str(e)


class GithubListStargazersBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        class StargazerItem(TypedDict):
            username: str
            url: str

        stargazer: StargazerItem = SchemaField(
            title="Stargazer",
            description="Stargazers with their username and profile URL",
        )
        error: str = SchemaField(
            description="Error message if listing stargazers failed"
        )

    def __init__(self):
        super().__init__(
            id="a4b9c2d1-e5f6-4g7h-8i9j-0k1l2m3n4o5p",  # Generated unique UUID
            description="This block lists all users who have starred a specified GitHub repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListStargazersBlock.Input,
            output_schema=GithubListStargazersBlock.Output,
            test_input={
                "repo_url": "https://github.com/owner/repo",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "stargazer",
                    {
                        "username": "octocat",
                        "url": "https://github.com/octocat",
                    },
                )
            ],
            test_mock={
                "list_stargazers": lambda *args, **kwargs: [
                    {
                        "username": "octocat",
                        "url": "https://github.com/octocat",
                    }
                ]
            },
        )

    @staticmethod
    def list_stargazers(
        credentials: GithubCredentials, repo_url: str
    ) -> list[Output.StargazerItem]:
        api = get_api(credentials)
        # Add /stargazers to the repo URL to get stargazers endpoint
        stargazers_url = f"{repo_url}/stargazers"
        # Set accept header to get starred_at timestamp
        headers = {"Accept": "application/vnd.github.star+json"}
        response = api.get(stargazers_url, headers=headers)
        data = response.json()

        stargazers: list[GithubListStargazersBlock.Output.StargazerItem] = [
            {
                "username": stargazer["login"],
                "url": stargazer["html_url"],
            }
            for stargazer in data
        ]
        return stargazers

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            stargazers = self.list_stargazers(
                credentials,
                input_data.repo_url,
            )
            yield from (("stargazer", stargazer) for stargazer in stargazers)
        except Exception as e:
            yield "error", str(e)
