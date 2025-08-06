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


class ReviewEvent(Enum):
    COMMENT = "COMMENT"
    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"


class GithubCreatePRReviewBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
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
            description="This block creates a review on a GitHub pull request. You can either create it as a draft or post it immediately.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreatePRReviewBlock.Input,
            output_schema=GithubCreatePRReviewBlock.Output,
            test_input={
                "pr_url": "https://github.com/owner/repo/pull/1",
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
        pr_url: str,
        body: str,
        event: ReviewEvent,
        create_as_draft: bool,
    ) -> tuple[int, str, str]:
        api = get_api(credentials)

        # Extract owner, repo, and PR number from URL
        # Format: https://github.com/owner/repo/pull/123
        parts = pr_url.split("/")
        owner = parts[3]
        repo = parts[4]
        pr_number = parts[6]

        # GitHub API endpoint for creating reviews
        reviews_url = f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"

        # Prepare the request data
        data = {
            "body": body,
            "event": event.value if not create_as_draft else "PENDING",
        }

        # Create the review
        response = await api.post(reviews_url, json=data)
        review_data = response.json()

        # If not a draft and event is not COMMENT, submit the review
        if not create_as_draft and event != ReviewEvent.COMMENT:
            submit_url = f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_data['id']}/events"
            submit_data = {"event": event.value}
            submit_response = await api.post(submit_url, json=submit_data)
            review_data = submit_response.json()

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
                input_data.pr_url,
                input_data.body,
                input_data.event,
                input_data.create_as_draft,
            )
            yield "review_id", review_id
            yield "state", state
            yield "html_url", html_url
        except Exception as e:
            yield "error", str(e)


class GithubListPRReviewsBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
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
                "pr_url": "https://github.com/owner/repo/pull/1",
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
        credentials: GithubCredentials, pr_url: str
    ) -> list[Output.ReviewItem]:
        api = get_api(credentials)

        # Extract owner, repo, and PR number from URL
        parts = pr_url.split("/")
        owner = parts[3]
        repo = parts[4]
        pr_number = parts[6]

        # GitHub API endpoint for listing reviews
        reviews_url = f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"

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
            input_data.pr_url,
        )
        yield "reviews", reviews
        for review in reviews:
            yield "review", review


class GithubSubmitPendingReviewBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
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
                "pr_url": "https://github.com/owner/repo/pull/1",
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
        pr_url: str,
        review_id: int,
        event: ReviewEvent,
    ) -> tuple[str, str]:
        api = get_api(credentials)

        # Extract owner, repo, and PR number from URL
        parts = pr_url.split("/")
        owner = parts[3]
        repo = parts[4]
        pr_number = parts[6]

        # GitHub API endpoint for submitting a review
        submit_url = (
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}/events"
        )

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
                input_data.pr_url,
                input_data.review_id,
                input_data.event,
            )
            yield "state", state
            yield "html_url", html_url
        except Exception as e:
            yield "error", str(e)


class GithubCreatePRReviewCommentBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
        )
        body: str = SchemaField(
            description="The comment text",
            placeholder="Enter your comment",
        )
        review_id: Optional[int] = SchemaField(
            description="ID of an existing draft review to add this comment to (optional)",
            placeholder="123456",
            default=None,
            advanced=True,
        )
        path: Optional[str] = SchemaField(
            description="The relative path to the file to comment on (for inline comments)",
            placeholder="src/main.py",
            default=None,
            advanced=True,
        )
        line: Optional[int] = SchemaField(
            description="The line number to comment on (deprecated, use start_line for new comments)",
            default=None,
            advanced=True,
        )
        start_line: Optional[int] = SchemaField(
            description="The first line of the range to comment on",
            default=None,
            advanced=True,
        )
        side: str = SchemaField(
            description="The side of the diff to comment on (LEFT or RIGHT)",
            default="RIGHT",
            advanced=True,
        )
        start_side: str = SchemaField(
            description="The side of the first line of the range (LEFT or RIGHT)",
            default="RIGHT",
            advanced=True,
        )

    class Output(BlockSchema):
        comment_id: int = SchemaField(description="ID of the created comment")
        html_url: str = SchemaField(description="URL of the created comment")
        error: str = SchemaField(
            description="Error message if the comment creation failed"
        )

    def __init__(self):
        super().__init__(
            id="4a5c6d8e-9f10-4b2a-8c3e-7d5a9b1e3f4c",
            description="This block creates a comment on a GitHub pull request. Can be standalone or part of a draft review.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubCreatePRReviewCommentBlock.Input,
            output_schema=GithubCreatePRReviewCommentBlock.Output,
            test_input={
                "pr_url": "https://github.com/owner/repo/pull/1",
                "body": "This looks good!",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("comment_id", 456789),
                (
                    "html_url",
                    "https://github.com/owner/repo/pull/1#issuecomment-456789",
                ),
            ],
            test_mock={
                "create_comment": lambda *args, **kwargs: (
                    456789,
                    "https://github.com/owner/repo/pull/1#issuecomment-456789",
                )
            },
        )

    @staticmethod
    async def create_comment(
        credentials: GithubCredentials,
        pr_url: str,
        body: str,
        review_id: Optional[int] = None,
        path: Optional[str] = None,
        line: Optional[int] = None,
        start_line: Optional[int] = None,
        side: str = "RIGHT",
        start_side: str = "RIGHT",
    ) -> tuple[int, str]:
        api = get_api(credentials)

        # Extract owner, repo, and PR number from URL
        parts = pr_url.split("/")
        owner = parts[3]
        repo = parts[4]
        pr_number = parts[6]

        # Prepare the comment data
        data: dict = {"body": body}

        # Add inline comment fields if provided
        if path:
            data["path"] = path
            if line is not None:
                data["line"] = line
            if start_line is not None:
                data["start_line"] = start_line
                data["line"] = (
                    line or start_line
                )  # line is required when start_line is provided
            if side:
                data["side"] = side
            if start_side and start_line is not None:
                data["start_side"] = start_side

        # Determine the endpoint based on whether this is part of a review
        if review_id:
            # Add comment to existing draft review
            comment_url = (
                f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}/comments"
            )
        else:
            # Create standalone comment
            comment_url = f"/repos/{owner}/{repo}/pulls/{pr_number}/comments"

        response = await api.post(comment_url, json=data)
        comment_data = response.json()

        return comment_data["id"], comment_data["html_url"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            comment_id, html_url = await self.create_comment(
                credentials,
                input_data.pr_url,
                input_data.body,
                input_data.review_id,
                input_data.path,
                input_data.line,
                input_data.start_line,
                input_data.side,
                input_data.start_side,
            )
            yield "comment_id", comment_id
            yield "html_url", html_url
        except Exception as e:
            yield "error", str(e)


class GithubResolveReviewDiscussionBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
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
                "pr_url": "https://github.com/owner/repo/pull/1",
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
        pr_url: str,
        comment_id: int,
        resolve: bool,
    ) -> bool:
        api = get_api(credentials)

        # Extract owner, repo, and PR number from URL
        parts = pr_url.split("/")
        owner = parts[3]
        repo = parts[4]
        pr_number = parts[6]

        # GitHub GraphQL API is needed for resolving/unresolving discussions
        # First, we need to get the node ID of the comment
        graphql_url = "/graphql"

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

        variables = {"owner": owner, "repo": repo, "number": int(pr_number)}

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
        mutation = (
            """
        mutation($threadId: ID!, $resolve: Boolean!) {
          resolveReviewThread(input: {threadId: $threadId, resolve: $resolve}) {
            thread {
              isResolved
            }
          }
        }
        """
            if resolve
            else """
        mutation($threadId: ID!) {
          unresolveReviewThread(input: {threadId: $threadId}) {
            thread {
              isResolved
            }
          }
        }
        """
        )

        mutation_variables = {"threadId": thread_id}
        if resolve:
            mutation_variables["resolve"] = True

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
                input_data.pr_url,
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
        pr_url: str = SchemaField(
            description="URL of the GitHub pull request",
            placeholder="https://github.com/owner/repo/pull/1",
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
                "pr_url": "https://github.com/owner/repo/pull/1",
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
        pr_url: str,
        review_id: Optional[int] = None,
    ) -> list[Output.CommentItem]:
        api = get_api(credentials)

        # Extract owner, repo, and PR number from URL
        parts = pr_url.split("/")
        owner = parts[3]
        repo = parts[4]
        pr_number = parts[6]

        # Determine the endpoint based on whether we want comments from a specific review
        if review_id:
            # Get comments from a specific review
            comments_url = (
                f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}/comments"
            )
        else:
            # Get all review comments on the PR
            comments_url = f"/repos/{owner}/{repo}/pulls/{pr_number}/comments"

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
                input_data.pr_url,
                input_data.review_id,
            )
            yield "comments", comments
            for comment in comments:
                yield "comment", comment
        except Exception as e:
            yield "error", str(e)
