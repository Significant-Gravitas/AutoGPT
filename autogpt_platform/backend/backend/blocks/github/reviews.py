import logging
from enum import Enum
from typing import Any, List, Optional

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


class ReviewEvent(Enum):
    COMMENT = "COMMENT"
    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"


class GithubCreatePRReviewBlock(Block):
    class Input(BlockSchema):
        class ReviewComment(TypedDict, total=False):
            path: str
            position: Optional[int]
            body: str
            line: Optional[int]  # Will be used as position if position not provided

        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description="GitHub repository",
            placeholder="owner/repo",
        )
        pr_number: int = SchemaField(
            description="Pull request number",
            placeholder="123",
        )
        body: str = SchemaField(
            description="Body of the review comment",
            placeholder="Enter your review comment",
        )
        event: ReviewEvent = SchemaField(
            description="The review action to perform",
            default=ReviewEvent.COMMENT,
        )
        create_as_draft: bool = SchemaField(
            description="Create the review as a draft (pending) or post it immediately",
            default=False,
            advanced=False,
        )
        comments: Optional[List[ReviewComment]] = SchemaField(
            description="Optional inline comments to add to specific files/lines. Note: Only path, body, and position are supported. Position is line number in diff from first @@ hunk.",
            default=None,
            advanced=True,
        )

    class Output(BlockSchema):
        review_id: int = SchemaField(description="ID of the created review")
        state: str = SchemaField(
            description="State of the review (e.g., PENDING, COMMENTED, APPROVED, CHANGES_REQUESTED)"
        )
        html_url: str = SchemaField(description="URL of the created review")
        error: str = SchemaField(
            description="Error message if the review creation failed"
        )

    def __init__(self):
        super().__init__(
            id="84754b30-97d2-4c37-a3b8-eb39f268275b",
            description="This block creates a review on a GitHub pull request with optional inline comments. You can create it as a draft or post immediately. Note: For inline comments, 'position' should be the line number in the diff (starting from the first @@ hunk header).",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreatePRReviewBlock.Input,
            output_schema=GithubCreatePRReviewBlock.Output,
            test_input={
                "repo": "owner/repo",
                "pr_number": 1,
                "body": "This looks good to me!",
                "event": "APPROVE",
                "create_as_draft": False,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("review_id", 123456),
                ("state", "APPROVED"),
                (
                    "html_url",
                    "https://github.com/owner/repo/pull/1#pullrequestreview-123456",
                ),
            ],
            test_mock={
                "create_review": lambda *args, **kwargs: (
                    123456,
                    "APPROVED",
                    "https://github.com/owner/repo/pull/1#pullrequestreview-123456",
                )
            },
        )

    @staticmethod
    async def create_review(
        credentials: GithubCredentials,
        repo: str,
        pr_number: int,
        body: str,
        event: ReviewEvent,
        create_as_draft: bool,
        comments: Optional[List[Input.ReviewComment]] = None,
    ) -> tuple[int, str, str]:
        api = get_api(credentials, convert_urls=False)

        # GitHub API endpoint for creating reviews
        reviews_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"

        # Get commit_id if we have comments
        commit_id = None
        if comments:
            # Get PR details to get the head commit for inline comments
            pr_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
            pr_response = await api.get(pr_url)
            pr_data = pr_response.json()
            commit_id = pr_data["head"]["sha"]

        # Prepare the request data
        # If create_as_draft is True, omit the event field (creates a PENDING review)
        # Otherwise, use the actual event value which will auto-submit the review
        data: dict[str, Any] = {"body": body}

        # Add commit_id if we have it
        if commit_id:
            data["commit_id"] = commit_id

        # Add comments if provided
        if comments:
            # Process comments to ensure they have the required fields
            processed_comments = []
            for comment in comments:
                comment_data: dict = {
                    "path": comment.get("path", ""),
                    "body": comment.get("body", ""),
                }
                # Add position or line
                # Note: For review comments, only position is supported (not line/side)
                if "position" in comment and comment.get("position") is not None:
                    comment_data["position"] = comment.get("position")
                elif "line" in comment and comment.get("line") is not None:
                    # Note: Using line as position - may not work correctly
                    # Position should be calculated from the diff
                    comment_data["position"] = comment.get("line")

                # Note: side, start_line, and start_side are NOT supported for review comments
                # They are only for standalone PR comments

                processed_comments.append(comment_data)

            data["comments"] = processed_comments

        if not create_as_draft:
            # Only add event field if not creating a draft
            data["event"] = event.value

        # Create the review
        response = await api.post(reviews_url, json=data)
        review_data = response.json()

        return review_data["id"], review_data["state"], review_data["html_url"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            review_id, state, html_url = await self.create_review(
                credentials,
                input_data.repo,
                input_data.pr_number,
                input_data.body,
                input_data.event,
                input_data.create_as_draft,
                input_data.comments,
            )
            yield "review_id", review_id
            yield "state", state
            yield "html_url", html_url
        except Exception as e:
            yield "error", str(e)


class GithubListPRReviewsBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description="GitHub repository",
            placeholder="owner/repo",
        )
        pr_number: int = SchemaField(
            description="Pull request number",
            placeholder="123",
        )

    class Output(BlockSchema):
        class ReviewItem(TypedDict):
            id: int
            user: str
            state: str
            body: str
            html_url: str

        review: ReviewItem = SchemaField(
            title="Review",
            description="Individual review with details",
        )
        reviews: list[ReviewItem] = SchemaField(
            description="List of all reviews on the pull request"
        )
        error: str = SchemaField(description="Error message if listing reviews failed")

    def __init__(self):
        super().__init__(
            id="f79bc6eb-33c0-4099-9c0f-d664ae1ba4d0",
            description="This block lists all reviews for a specified GitHub pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListPRReviewsBlock.Input,
            output_schema=GithubListPRReviewsBlock.Output,
            test_input={
                "repo": "owner/repo",
                "pr_number": 1,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "reviews",
                    [
                        {
                            "id": 123456,
                            "user": "reviewer1",
                            "state": "APPROVED",
                            "body": "Looks good!",
                            "html_url": "https://github.com/owner/repo/pull/1#pullrequestreview-123456",
                        }
                    ],
                ),
                (
                    "review",
                    {
                        "id": 123456,
                        "user": "reviewer1",
                        "state": "APPROVED",
                        "body": "Looks good!",
                        "html_url": "https://github.com/owner/repo/pull/1#pullrequestreview-123456",
                    },
                ),
            ],
            test_mock={
                "list_reviews": lambda *args, **kwargs: [
                    {
                        "id": 123456,
                        "user": "reviewer1",
                        "state": "APPROVED",
                        "body": "Looks good!",
                        "html_url": "https://github.com/owner/repo/pull/1#pullrequestreview-123456",
                    }
                ]
            },
        )

    @staticmethod
    async def list_reviews(
        credentials: GithubCredentials, repo: str, pr_number: int
    ) -> list[Output.ReviewItem]:
        api = get_api(credentials, convert_urls=False)

        # GitHub API endpoint for listing reviews
        reviews_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"

        response = await api.get(reviews_url)
        data = response.json()

        reviews: list[GithubListPRReviewsBlock.Output.ReviewItem] = [
            {
                "id": review["id"],
                "user": review["user"]["login"],
                "state": review["state"],
                "body": review.get("body", ""),
                "html_url": review["html_url"],
            }
            for review in data
        ]
        return reviews

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        reviews = await self.list_reviews(
            credentials,
            input_data.repo,
            input_data.pr_number,
        )
        yield "reviews", reviews
        for review in reviews:
            yield "review", review


class GithubSubmitPendingReviewBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description="GitHub repository",
            placeholder="owner/repo",
        )
        pr_number: int = SchemaField(
            description="Pull request number",
            placeholder="123",
        )
        review_id: int = SchemaField(
            description="ID of the pending review to submit",
            placeholder="123456",
        )
        event: ReviewEvent = SchemaField(
            description="The review action to perform when submitting",
            default=ReviewEvent.COMMENT,
        )

    class Output(BlockSchema):
        state: str = SchemaField(description="State of the submitted review")
        html_url: str = SchemaField(description="URL of the submitted review")
        error: str = SchemaField(
            description="Error message if the review submission failed"
        )

    def __init__(self):
        super().__init__(
            id="2e468217-7ca0-4201-9553-36e93eb9357a",
            description="This block submits a pending (draft) review on a GitHub pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubSubmitPendingReviewBlock.Input,
            output_schema=GithubSubmitPendingReviewBlock.Output,
            test_input={
                "repo": "owner/repo",
                "pr_number": 1,
                "review_id": 123456,
                "event": "APPROVE",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("state", "APPROVED"),
                (
                    "html_url",
                    "https://github.com/owner/repo/pull/1#pullrequestreview-123456",
                ),
            ],
            test_mock={
                "submit_review": lambda *args, **kwargs: (
                    "APPROVED",
                    "https://github.com/owner/repo/pull/1#pullrequestreview-123456",
                )
            },
        )

    @staticmethod
    async def submit_review(
        credentials: GithubCredentials,
        repo: str,
        pr_number: int,
        review_id: int,
        event: ReviewEvent,
    ) -> tuple[str, str]:
        api = get_api(credentials, convert_urls=False)

        # GitHub API endpoint for submitting a review
        submit_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews/{review_id}/events"

        data = {"event": event.value}

        response = await api.post(submit_url, json=data)
        review_data = response.json()

        return review_data["state"], review_data["html_url"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            state, html_url = await self.submit_review(
                credentials,
                input_data.repo,
                input_data.pr_number,
                input_data.review_id,
                input_data.event,
            )
            yield "state", state
            yield "html_url", html_url
        except Exception as e:
            yield "error", str(e)


class GithubResolveReviewDiscussionBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description="GitHub repository",
            placeholder="owner/repo",
        )
        pr_number: int = SchemaField(
            description="Pull request number",
            placeholder="123",
        )
        comment_id: int = SchemaField(
            description="ID of the review comment to resolve/unresolve",
            placeholder="123456",
        )
        resolve: bool = SchemaField(
            description="Whether to resolve (true) or unresolve (false) the discussion",
            default=True,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the operation was successful")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="b4b8a38c-95ae-4c91-9ef8-c2cffaf2b5d1",
            description="This block resolves or unresolves a review discussion thread on a GitHub pull request.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubResolveReviewDiscussionBlock.Input,
            output_schema=GithubResolveReviewDiscussionBlock.Output,
            test_input={
                "repo": "owner/repo",
                "pr_number": 1,
                "comment_id": 123456,
                "resolve": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"resolve_discussion": lambda *args, **kwargs: True},
        )

    @staticmethod
    async def resolve_discussion(
        credentials: GithubCredentials,
        repo: str,
        pr_number: int,
        comment_id: int,
        resolve: bool,
    ) -> bool:
        api = get_api(credentials, convert_urls=False)

        # Extract owner and repo name
        parts = repo.split("/")
        owner = parts[0]
        repo_name = parts[1]

        # GitHub GraphQL API is needed for resolving/unresolving discussions
        # First, we need to get the node ID of the comment
        graphql_url = "https://api.github.com/graphql"

        # Query to get the review comment node ID
        query = """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $number) {
              reviewThreads(first: 100) {
                nodes {
                  comments(first: 100) {
                    nodes {
                      databaseId
                      id
                    }
                  }
                  id
                  isResolved
                }
              }
            }
          }
        }
        """

        variables = {"owner": owner, "repo": repo_name, "number": pr_number}

        response = await api.post(
            graphql_url, json={"query": query, "variables": variables}
        )
        data = response.json()

        # Find the thread containing our comment
        thread_id = None
        for thread in data["data"]["repository"]["pullRequest"]["reviewThreads"][
            "nodes"
        ]:
            for comment in thread["comments"]["nodes"]:
                if comment["databaseId"] == comment_id:
                    thread_id = thread["id"]
                    break
            if thread_id:
                break

        if not thread_id:
            raise ValueError(f"Comment {comment_id} not found in pull request")

        # Now resolve or unresolve the thread
        # GitHub's GraphQL API has separate mutations for resolve and unresolve
        if resolve:
            mutation = """
            mutation($threadId: ID!) {
              resolveReviewThread(input: {threadId: $threadId}) {
                thread {
                  isResolved
                }
              }
            }
            """
        else:
            mutation = """
            mutation($threadId: ID!) {
              unresolveReviewThread(input: {threadId: $threadId}) {
                thread {
                  isResolved
                }
              }
            }
            """

        mutation_variables = {"threadId": thread_id}

        response = await api.post(
            graphql_url, json={"query": mutation, "variables": mutation_variables}
        )
        result = response.json()

        if "errors" in result:
            raise Exception(f"GraphQL error: {result['errors']}")

        return True

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            success = await self.resolve_discussion(
                credentials,
                input_data.repo,
                input_data.pr_number,
                input_data.comment_id,
                input_data.resolve,
            )
            yield "success", success
        except Exception as e:
            yield "success", False
            yield "error", str(e)


class GithubGetPRReviewCommentsBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description="GitHub repository",
            placeholder="owner/repo",
        )
        pr_number: int = SchemaField(
            description="Pull request number",
            placeholder="123",
        )
        review_id: Optional[int] = SchemaField(
            description="ID of a specific review to get comments from (optional)",
            placeholder="123456",
            default=None,
            advanced=True,
        )

    class Output(BlockSchema):
        class CommentItem(TypedDict):
            id: int
            user: str
            body: str
            path: str
            line: int
            side: str
            created_at: str
            updated_at: str
            in_reply_to_id: Optional[int]
            html_url: str

        comment: CommentItem = SchemaField(
            title="Comment",
            description="Individual review comment with details",
        )
        comments: list[CommentItem] = SchemaField(
            description="List of all review comments on the pull request"
        )
        error: str = SchemaField(description="Error message if getting comments failed")

    def __init__(self):
        super().__init__(
            id="1d34db7f-10c1-45c1-9d43-749f743c8bd4",
            description="This block gets all review comments from a GitHub pull request or from a specific review.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubGetPRReviewCommentsBlock.Input,
            output_schema=GithubGetPRReviewCommentsBlock.Output,
            test_input={
                "repo": "owner/repo",
                "pr_number": 1,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "comments",
                    [
                        {
                            "id": 123456,
                            "user": "reviewer1",
                            "body": "This needs improvement",
                            "path": "src/main.py",
                            "line": 42,
                            "side": "RIGHT",
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-01T00:00:00Z",
                            "in_reply_to_id": None,
                            "html_url": "https://github.com/owner/repo/pull/1#discussion_r123456",
                        }
                    ],
                ),
                (
                    "comment",
                    {
                        "id": 123456,
                        "user": "reviewer1",
                        "body": "This needs improvement",
                        "path": "src/main.py",
                        "line": 42,
                        "side": "RIGHT",
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-01T00:00:00Z",
                        "in_reply_to_id": None,
                        "html_url": "https://github.com/owner/repo/pull/1#discussion_r123456",
                    },
                ),
            ],
            test_mock={
                "get_comments": lambda *args, **kwargs: [
                    {
                        "id": 123456,
                        "user": "reviewer1",
                        "body": "This needs improvement",
                        "path": "src/main.py",
                        "line": 42,
                        "side": "RIGHT",
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-01T00:00:00Z",
                        "in_reply_to_id": None,
                        "html_url": "https://github.com/owner/repo/pull/1#discussion_r123456",
                    }
                ]
            },
        )

    @staticmethod
    async def get_comments(
        credentials: GithubCredentials,
        repo: str,
        pr_number: int,
        review_id: Optional[int] = None,
    ) -> list[Output.CommentItem]:
        api = get_api(credentials, convert_urls=False)

        # Determine the endpoint based on whether we want comments from a specific review
        if review_id:
            # Get comments from a specific review
            comments_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews/{review_id}/comments"
        else:
            # Get all review comments on the PR
            comments_url = (
                f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
            )

        response = await api.get(comments_url)
        data = response.json()

        comments: list[GithubGetPRReviewCommentsBlock.Output.CommentItem] = [
            {
                "id": comment["id"],
                "user": comment["user"]["login"],
                "body": comment["body"],
                "path": comment.get("path", ""),
                "line": comment.get("line", 0),
                "side": comment.get("side", ""),
                "created_at": comment["created_at"],
                "updated_at": comment["updated_at"],
                "in_reply_to_id": comment.get("in_reply_to_id"),
                "html_url": comment["html_url"],
            }
            for comment in data
        ]
        return comments

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            comments = await self.get_comments(
                credentials,
                input_data.repo,
                input_data.pr_number,
                input_data.review_id,
            )
            yield "comments", comments
            for comment in comments:
                yield "comment", comment
        except Exception as e:
            yield "error", str(e)


class GithubCreateCommentObjectBlock(Block):
    class Input(BlockSchema):
        path: str = SchemaField(
            description="The file path to comment on",
            placeholder="src/main.py",
        )
        body: str = SchemaField(
            description="The comment text",
            placeholder="Please fix this issue",
        )
        position: Optional[int] = SchemaField(
            description="Position in the diff (line number from first @@ hunk). Use this OR line.",
            placeholder="6",
            default=None,
            advanced=True,
        )
        line: Optional[int] = SchemaField(
            description="Line number in the file (will be used as position if position not provided)",
            placeholder="42",
            default=None,
            advanced=True,
        )
        side: Optional[str] = SchemaField(
            description="Side of the diff to comment on (NOTE: Only for standalone comments, not review comments)",
            default="RIGHT",
            advanced=True,
        )
        start_line: Optional[int] = SchemaField(
            description="Start line for multi-line comments (NOTE: Only for standalone comments, not review comments)",
            default=None,
            advanced=True,
        )
        start_side: Optional[str] = SchemaField(
            description="Side for the start of multi-line comments (NOTE: Only for standalone comments, not review comments)",
            default=None,
            advanced=True,
        )

    class Output(BlockSchema):
        comment_object: dict = SchemaField(
            description="The comment object formatted for GitHub API"
        )

    def __init__(self):
        super().__init__(
            id="b7d5e4f2-8c3a-4e6b-9f1d-7a8b9c5e4d3f",
            description="Creates a comment object for use with GitHub blocks. Note: For review comments, only path, body, and position are used. Side fields are only for standalone PR comments.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreateCommentObjectBlock.Input,
            output_schema=GithubCreateCommentObjectBlock.Output,
            test_input={
                "path": "src/main.py",
                "body": "Please fix this issue",
                "position": 6,
            },
            test_output=[
                (
                    "comment_object",
                    {
                        "path": "src/main.py",
                        "body": "Please fix this issue",
                        "position": 6,
                    },
                ),
            ],
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        # Build the comment object
        comment_obj: dict = {
            "path": input_data.path,
            "body": input_data.body,
        }

        # Add position or line
        if input_data.position is not None:
            comment_obj["position"] = input_data.position
        elif input_data.line is not None:
            # Note: line will be used as position, which may not be accurate
            # Position should be calculated from the diff
            comment_obj["position"] = input_data.line

        # Add optional fields only if they differ from defaults or are explicitly provided
        if input_data.side and input_data.side != "RIGHT":
            comment_obj["side"] = input_data.side
        if input_data.start_line is not None:
            comment_obj["start_line"] = input_data.start_line
        if input_data.start_side:
            comment_obj["start_side"] = input_data.start_side

        yield "comment_object", comment_obj
