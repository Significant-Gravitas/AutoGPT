import logging
from urllib.parse import urlparse

from typing_extensions import TypedDict

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

from ._api import convert_comment_url_to_api_endpoint, get_api
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)

logger = logging.getLogger(__name__)


def is_github_url(url: str) -> bool:
    return urlparse(url).netloc == "github.com"


# --8<-- [start:GithubCommentBlockExample]
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
        id: int = SchemaField(description="ID of the created comment")
        url: str = SchemaField(description="URL to the comment on GitHub")
        error: str = SchemaField(
            description="Error message if the comment posting failed"
        )

    def __init__(self):
        super().__init__(
            id="a8db4d8d-db1c-4a25-a1b0-416a8c33602b",
            description="This block posts a comment on a specified GitHub issue or pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCommentBlock.Input,
            output_schema=GithubCommentBlock.Output,
            test_input=[
                {
                    "issue_url": "https://github.com/owner/repo/issues/1",
                    "comment": "This is a test comment.",
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
                {
                    "issue_url": "https://github.com/owner/repo/pull/1",
                    "comment": "This is a test comment.",
                    "credentials": TEST_CREDENTIALS_INPUT,
                },
            ],
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", 1337),
                ("url", "https://github.com/owner/repo/issues/1#issuecomment-1337"),
                ("id", 1337),
                (
                    "url",
                    "https://github.com/owner/repo/issues/1#issuecomment-1337",
                ),
            ],
            test_mock={
                "post_comment": lambda *args, **kwargs: (
                    1337,
                    "https://github.com/owner/repo/issues/1#issuecomment-1337",
                )
            },
        )

    @staticmethod
    def post_comment(
        credentials: GithubCredentials, issue_url: str, body_text: str
    ) -> tuple[int, str]:
        api = get_api(credentials)
        data = {"body": body_text}
        if "pull" in issue_url:
            issue_url = issue_url.replace("pull", "issues")
        comments_url = issue_url + "/comments"
        response = api.post(comments_url, json=data)
        comment = response.json()
        return comment["id"], comment["html_url"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        id, url = self.post_comment(
            credentials,
            input_data.issue_url,
            input_data.comment,
        )
        yield "id", id
        yield "url", url


# --8<-- [end:GithubCommentBlockExample]


class GithubUpdateCommentBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        comment_url: str = SchemaField(
            description="URL of the GitHub comment",
            placeholder="https://github.com/owner/repo/issues/1#issuecomment-123456789",
            default="",
            advanced=False,
        )
        issue_url: str = SchemaField(
            description="URL of the GitHub issue or pull request",
            placeholder="https://github.com/owner/repo/issues/1",
            default="",
        )
        comment_id: str = SchemaField(
            description="ID of the GitHub comment",
            placeholder="123456789",
            default="",
        )
        comment: str = SchemaField(
            description="Comment to update",
            placeholder="Enter your comment",
        )

    class Output(BlockSchema):
        id: int = SchemaField(description="ID of the updated comment")
        url: str = SchemaField(description="URL to the comment on GitHub")
        error: str = SchemaField(
            description="Error message if the comment update failed"
        )

    def __init__(self):
        super().__init__(
            id="b3f4d747-10e3-4e69-8c51-f2be1d99c9a7",
            description="This block updates a comment on a specified GitHub issue or pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubUpdateCommentBlock.Input,
            output_schema=GithubUpdateCommentBlock.Output,
            test_input={
                "comment_url": "https://github.com/owner/repo/issues/1#issuecomment-123456789",
                "comment": "This is an updated comment.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", 123456789),
                (
                    "url",
                    "https://github.com/owner/repo/issues/1#issuecomment-123456789",
                ),
            ],
            test_mock={
                "update_comment": lambda *args, **kwargs: (
                    123456789,
                    "https://github.com/owner/repo/issues/1#issuecomment-123456789",
                )
            },
        )

    @staticmethod
    def update_comment(
        credentials: GithubCredentials, comment_url: str, body_text: str
    ) -> tuple[int, str]:
        api = get_api(credentials, convert_urls=False)
        data = {"body": body_text}
        url = convert_comment_url_to_api_endpoint(comment_url)

        logger.info(url)
        response = api.patch(url, json=data)
        comment = response.json()
        return comment["id"], comment["html_url"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        if (
            not input_data.comment_url
            and input_data.comment_id
            and input_data.issue_url
        ):
            parsed_url = urlparse(input_data.issue_url)
            path_parts = parsed_url.path.strip("/").split("/")
            owner, repo = path_parts[0], path_parts[1]

            input_data.comment_url = f"https://api.github.com/repos/{owner}/{repo}/issues/comments/{input_data.comment_id}"

        elif (
            not input_data.comment_url
            and not input_data.comment_id
            and input_data.issue_url
        ):
            raise ValueError(
                "Must provide either comment_url or comment_id and issue_url"
            )
        id, url = self.update_comment(
            credentials,
            input_data.comment_url,
            input_data.comment,
        )
        yield "id", id
        yield "url", url


class GithubListCommentsBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        issue_url: str = SchemaField(
            description="URL of the GitHub issue or pull request",
            placeholder="https://github.com/owner/repo/issues/1",
        )

    class Output(BlockSchema):
        class CommentItem(TypedDict):
            id: int
            body: str
            user: str
            url: str

        comment: CommentItem = SchemaField(
            title="Comment", description="Comments with their ID, body, user, and URL"
        )
        comments: list[CommentItem] = SchemaField(
            description="List of comments with their ID, body, user, and URL"
        )
        error: str = SchemaField(description="Error message if listing comments failed")

    def __init__(self):
        super().__init__(
            id="c4b5fb63-0005-4a11-b35a-0c2467bd6b59",
            description="This block lists all comments for a specified GitHub issue or pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListCommentsBlock.Input,
            output_schema=GithubListCommentsBlock.Output,
            test_input={
                "issue_url": "https://github.com/owner/repo/issues/1",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "comment",
                    {
                        "id": 123456789,
                        "body": "This is a test comment.",
                        "user": "test_user",
                        "url": "https://github.com/owner/repo/issues/1#issuecomment-123456789",
                    },
                ),
                (
                    "comments",
                    [
                        {
                            "id": 123456789,
                            "body": "This is a test comment.",
                            "user": "test_user",
                            "url": "https://github.com/owner/repo/issues/1#issuecomment-123456789",
                        }
                    ],
                ),
            ],
            test_mock={
                "list_comments": lambda *args, **kwargs: [
                    {
                        "id": 123456789,
                        "body": "This is a test comment.",
                        "user": "test_user",
                        "url": "https://github.com/owner/repo/issues/1#issuecomment-123456789",
                    }
                ]
            },
        )

    @staticmethod
    def list_comments(
        credentials: GithubCredentials, issue_url: str
    ) -> list[Output.CommentItem]:
        parsed_url = urlparse(issue_url)
        path_parts = parsed_url.path.strip("/").split("/")

        owner = path_parts[0]
        repo = path_parts[1]

        # GitHub API uses 'issues' for both issues and pull requests when it comes to comments
        issue_number = path_parts[3]  # Whether 'issues/123' or 'pull/123'

        # Construct the proper API URL directly
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"

        # Set convert_urls=False since we're already providing an API URL
        api = get_api(credentials, convert_urls=False)
        response = api.get(api_url)
        comments = response.json()
        parsed_comments: list[GithubListCommentsBlock.Output.CommentItem] = [
            {
                "id": comment["id"],
                "body": comment["body"],
                "user": comment["user"]["login"],
                "url": comment["html_url"],
            }
            for comment in comments
        ]
        return parsed_comments

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        comments = self.list_comments(
            credentials,
            input_data.issue_url,
        )
        yield from (("comment", comment) for comment in comments)
        yield "comments", comments


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
        number: int = SchemaField(description="Number of the created issue")
        url: str = SchemaField(description="URL of the created issue")
        error: str = SchemaField(
            description="Error message if the issue creation failed"
        )

    def __init__(self):
        super().__init__(
            id="691dad47-f494-44c3-a1e8-05b7990f2dab",
            description="This block creates a new issue on a specified GitHub repository.",
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
            test_output=[
                ("number", 1),
                ("url", "https://github.com/owner/repo/issues/1"),
            ],
            test_mock={
                "create_issue": lambda *args, **kwargs: (
                    1,
                    "https://github.com/owner/repo/issues/1",
                )
            },
        )

    @staticmethod
    def create_issue(
        credentials: GithubCredentials, repo_url: str, title: str, body: str
    ) -> tuple[int, str]:
        api = get_api(credentials)
        data = {"title": title, "body": body}
        issues_url = repo_url + "/issues"
        response = api.post(issues_url, json=data)
        issue = response.json()
        return issue["number"], issue["html_url"]

    def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        number, url = self.create_issue(
            credentials,
            input_data.repo_url,
            input_data.title,
            input_data.body,
        )
        yield "number", number
        yield "url", url


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
            id="6443c75d-032a-4772-9c08-230c707c8acc",
            description="This block reads the body, title, and user of a specified GitHub issue.",
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
        api = get_api(credentials)
        response = api.get(issue_url)
        data = response.json()
        title = data.get("title", "No title found")
        body = data.get("body", "No body content found")
        user = data.get("user", {}).get("login", "No user found")
        return title, body, user

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
        if title:
            yield "title", title
        if body:
            yield "body", body
        if user:
            yield "user", user


class GithubListIssuesBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo_url: str = SchemaField(
            description="URL of the GitHub repository",
            placeholder="https://github.com/owner/repo",
        )

    class Output(BlockSchema):
        class IssueItem(TypedDict):
            title: str
            url: str

        issue: IssueItem = SchemaField(
            title="Issue", description="Issues with their title and URL"
        )
        error: str = SchemaField(description="Error message if listing issues failed")

    def __init__(self):
        super().__init__(
            id="c215bfd7-0e57-4573-8f8c-f7d4963dcd74",
            description="This block lists all issues for a specified GitHub repository.",
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
                    "issue",
                    {
                        "title": "Issue 1",
                        "url": "https://github.com/owner/repo/issues/1",
                    },
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
    ) -> list[Output.IssueItem]:
        api = get_api(credentials)
        issues_url = repo_url + "/issues"
        response = api.get(issues_url)
        data = response.json()
        issues: list[GithubListIssuesBlock.Output.IssueItem] = [
            {"title": issue["title"], "url": issue["html_url"]} for issue in data
        ]
        return issues

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
        yield from (("issue", issue) for issue in issues)


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
            id="98bd6b77-9506-43d5-b669-6b9733c4b1f1",
            description="This block adds a label to a specified GitHub issue or pull request.",
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
        api = get_api(credentials)
        data = {"labels": [label]}
        labels_url = issue_url + "/labels"
        api.post(labels_url, json=data)
        return "Label added successfully"

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
        yield "status", status


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
            id="78f050c5-3e3a-48c0-9e5b-ef1ceca5589c",
            description="This block removes a label from a specified GitHub issue or pull request.",
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
        api = get_api(credentials)
        label_url = issue_url + f"/labels/{label}"
        api.delete(label_url)
        return "Label removed successfully"

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
        yield "status", status


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
            id="90507c72-b0ff-413a-886a-23bbbd66f542",
            description="This block assigns a user to a specified GitHub issue.",
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
        api = get_api(credentials)
        assignees_url = issue_url + "/assignees"
        data = {"assignees": [assignee]}
        api.post(assignees_url, json=data)
        return "Issue assigned successfully"

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
        yield "status", status


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
            id="d154002a-38f4-46c2-962d-2488f2b05ece",
            description="This block unassigns a user from a specified GitHub issue.",
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
        api = get_api(credentials)
        assignees_url = issue_url + "/assignees"
        data = {"assignees": [assignee]}
        api.delete(assignees_url, json=data)
        return "Issue unassigned successfully"

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
        yield "status", status
