import requests
from typing_extensions import TypedDict

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

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

        pull_request: PRItem = SchemaField(description="PRs with their title and URL")
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
        try:
            api_url = repo_url.replace("github.com", "api.github.com/repos") + "/pulls"
            headers = {
                "Authorization": credentials.bearer(),
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            pull_requests: list[GithubListPullRequestsBlock.Output.PRItem] = [
                {"title": pr["title"], "url": pr["html_url"]} for pr in data
            ]

            return pull_requests
        except Exception as e:
            return [{"title": "Error", "url": f"Failed to list issues: {str(e)}"}]

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
        if any("Failed" in pr["url"] for pr in pull_requests):
            yield "error", pull_requests[0]["url"]
        else:
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
        user: str = SchemaField(description="User who created the pull request")
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
            description="Reviewers with their username and profile URL"
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
            reviewers: list[GithubListPRReviewersBlock.Output.ReviewerItem] = [
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
            yield from (("reviewer", reviewer) for reviewer in reviewers)
