import logging
from datetime import datetime, timezone
from typing import Iterator, Literal

import praw
from praw.models import MoreComments
from pydantic import BaseModel, SecretStr

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    OAuth2Credentials,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.mock import MockObject
from backend.util.settings import Settings

RedditCredentials = OAuth2Credentials
RedditCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.REDDIT],
    Literal["oauth2"],
]


def RedditCredentialsField() -> RedditCredentialsInput:
    """Creates a Reddit credentials input on a block."""
    return CredentialsField(
        description="Connect your Reddit account to access Reddit features.",
    )


TEST_CREDENTIALS = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="reddit",
    access_token=SecretStr("mock-reddit-access-token"),
    refresh_token=SecretStr("mock-reddit-refresh-token"),
    access_token_expires_at=9999999999,
    scopes=["identity", "read", "submit", "history"],
    title="Mock Reddit credentials",
    username="mock-reddit-username",
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class RedditPost(BaseModel):
    id: str
    subreddit: str
    title: str
    body: str


class RedditComment(BaseModel):
    post_id: str
    comment: str


settings = Settings()
logger = logging.getLogger(__name__)


def get_praw(creds: RedditCredentials) -> praw.Reddit:
    """
    Create a PRAW Reddit client using OAuth2 credentials.

    Uses the refresh_token for authentication, which allows the client
    to automatically refresh the access token when needed.
    """
    client = praw.Reddit(
        client_id=settings.secrets.reddit_client_id,
        client_secret=settings.secrets.reddit_client_secret,
        refresh_token=(
            creds.refresh_token.get_secret_value() if creds.refresh_token else None
        ),
        user_agent=settings.config.reddit_user_agent,
    )
    me = client.user.me()
    if not me:
        raise ValueError("Invalid Reddit credentials.")
    logger.info(f"Logged in as Reddit user: {me.name}")
    return client


class GetRedditPostsBlock(Block):
    class Input(BlockSchemaInput):
        subreddit: str = SchemaField(
            description="Subreddit name, excluding the /r/ prefix",
            default="writingprompts",
        )
        credentials: RedditCredentialsInput = RedditCredentialsField()
        last_minutes: int | None = SchemaField(
            description="Post time to stop minutes ago while fetching posts",
            default=None,
        )
        last_post: str | None = SchemaField(
            description="Post ID to stop when reached while fetching posts",
            default=None,
        )
        post_limit: int | None = SchemaField(
            description="Number of posts to fetch", default=10
        )

    class Output(BlockSchemaOutput):
        post: RedditPost = SchemaField(description="Reddit post")
        posts: list[RedditPost] = SchemaField(description="List of all Reddit posts")

    def __init__(self):
        super().__init__(
            id="c6731acb-4285-4ee1-bc9b-03d0766c370f",
            description="This block fetches Reddit posts from a defined subreddit name.",
            categories={BlockCategory.SOCIAL},
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            input_schema=GetRedditPostsBlock.Input,
            output_schema=GetRedditPostsBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subreddit": "subreddit",
                "last_post": "id3",
                "post_limit": 2,
            },
            test_output=[
                (
                    "post",
                    RedditPost(
                        id="id1", subreddit="subreddit", title="title1", body="body1"
                    ),
                ),
                (
                    "post",
                    RedditPost(
                        id="id2", subreddit="subreddit", title="title2", body="body2"
                    ),
                ),
                (
                    "posts",
                    [
                        RedditPost(
                            id="id1",
                            subreddit="subreddit",
                            title="title1",
                            body="body1",
                        ),
                        RedditPost(
                            id="id2",
                            subreddit="subreddit",
                            title="title2",
                            body="body2",
                        ),
                    ],
                ),
            ],
            test_mock={
                "get_posts": lambda input_data, credentials: [
                    MockObject(id="id1", title="title1", selftext="body1"),
                    MockObject(id="id2", title="title2", selftext="body2"),
                    MockObject(id="id3", title="title2", selftext="body2"),
                ]
            },
        )

    @staticmethod
    def get_posts(
        input_data: Input, *, credentials: RedditCredentials
    ) -> Iterator[praw.reddit.Submission]:
        client = get_praw(credentials)
        subreddit = client.subreddit(input_data.subreddit)
        return subreddit.new(limit=input_data.post_limit or 10)

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        current_time = datetime.now(tz=timezone.utc)
        all_posts = []
        for post in self.get_posts(input_data=input_data, credentials=credentials):
            if input_data.last_minutes:
                post_datetime = datetime.fromtimestamp(
                    post.created_utc, tz=timezone.utc
                )
                time_difference = current_time - post_datetime
                if time_difference.total_seconds() / 60 > input_data.last_minutes:
                    continue

            if input_data.last_post and post.id == input_data.last_post:
                break

            reddit_post = RedditPost(
                id=post.id,
                subreddit=input_data.subreddit,
                title=post.title,
                body=post.selftext,
            )
            all_posts.append(reddit_post)
            yield "post", reddit_post

        yield "posts", all_posts


class PostRedditCommentBlock(Block):
    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        post_id: str = SchemaField(
            description="The ID of the post to comment on",
        )
        comment: str = SchemaField(
            description="The content of the comment to post",
        )

    class Output(BlockSchemaOutput):
        comment_id: str = SchemaField(description="Posted comment ID")

    def __init__(self):
        super().__init__(
            id="4a92261b-701e-4ffb-8970-675fd28e261f",
            description="This block posts a Reddit comment on a specified Reddit post.",
            categories={BlockCategory.SOCIAL},
            input_schema=PostRedditCommentBlock.Input,
            output_schema=PostRedditCommentBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "data": {"post_id": "id", "comment": "comment"},
            },
            test_output=[("comment_id", "dummy_comment_id")],
            test_mock={"reply_post": lambda creds, comment: "dummy_comment_id"},
        )

    @staticmethod
    def reply_post(creds: RedditCredentials, post_id: str, comment: str) -> str:
        client = get_praw(creds)
        submission = client.submission(id=post_id)
        new_comment = submission.reply(comment)
        if not new_comment:
            raise ValueError("Failed to post comment.")
        return new_comment.id

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        yield "comment_id", self.reply_post(
            credentials,
            post_id=input_data.post_id,
            comment=input_data.comment,
        )


class CreateRedditPostBlock(Block):
    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        subreddit: str = SchemaField(
            description="Subreddit to post to, excluding the /r/ prefix",
        )
        title: str = SchemaField(
            description="Title of the post",
        )
        content: str = SchemaField(
            description="Body text of the post (for text posts)",
            default="",
        )
        url: str | None = SchemaField(
            description="URL to submit (for link posts). If provided, content is ignored.",
            default=None,
        )
        flair_id: str | None = SchemaField(
            description="Flair ID to apply to the post",
            default=None,
        )

    class Output(BlockSchemaOutput):
        post_id: str = SchemaField(description="ID of the created post")
        post_url: str = SchemaField(description="URL of the created post")

    def __init__(self):
        super().__init__(
            id="f3a2b1c0-8d7e-4f6a-9b5c-1234567890ab",
            description="Create a new post on a subreddit. Can create text posts or link posts.",
            categories={BlockCategory.SOCIAL},
            input_schema=CreateRedditPostBlock.Input,
            output_schema=CreateRedditPostBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subreddit": "test",
                "title": "Test Post",
                "content": "This is a test post body.",
            },
            test_output=[
                ("post_id", "abc123"),
                ("post_url", "https://reddit.com/r/test/comments/abc123/test_post/"),
            ],
            test_mock={
                "create_post": lambda creds, subreddit, title, content, url, flair_id: (
                    "abc123",
                    "https://reddit.com/r/test/comments/abc123/test_post/",
                )
            },
        )

    @staticmethod
    def create_post(
        creds: RedditCredentials,
        subreddit: str,
        title: str,
        content: str = "",
        url: str | None = None,
        flair_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Create a new post on a subreddit.

        Args:
            creds: Reddit OAuth2 credentials
            subreddit: Subreddit name (without /r/ prefix)
            title: Post title
            content: Post body text (for text posts)
            url: URL to submit (for link posts, overrides content)
            flair_id: Optional flair ID to apply

        Returns:
            Tuple of (post_id, post_url)
        """
        client = get_praw(creds)
        sub = client.subreddit(subreddit)

        if url:
            submission = sub.submit(title=title, url=url, flair_id=flair_id)
        else:
            submission = sub.submit(title=title, selftext=content, flair_id=flair_id)

        return submission.id, f"https://reddit.com{submission.permalink}"

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        post_id, post_url = self.create_post(
            credentials,
            input_data.subreddit,
            input_data.title,
            input_data.content,
            input_data.url,
            input_data.flair_id,
        )
        yield "post_id", post_id
        yield "post_url", post_url


class RedditPostDetails(BaseModel):
    """Detailed information about a Reddit post."""

    id: str
    subreddit: str
    title: str
    body: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    url: str
    permalink: str
    is_self: bool
    over_18: bool


class GetRedditPostBlock(Block):
    """Get detailed information about a specific Reddit post."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        post_id: str = SchemaField(
            description="The ID of the post to fetch (e.g., 'abc123' or full ID 't3_abc123')",
        )

    class Output(BlockSchemaOutput):
        post: RedditPostDetails = SchemaField(description="Detailed post information")
        error: str = SchemaField(
            description="Error message if the post couldn't be fetched"
        )

    def __init__(self):
        super().__init__(
            id="36e6a259-168c-4032-83ec-b2935d0e4584",
            description="Get detailed information about a specific Reddit post by its ID.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetRedditPostBlock.Input,
            output_schema=GetRedditPostBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "post_id": "abc123",
            },
            test_output=[
                (
                    "post",
                    RedditPostDetails(
                        id="abc123",
                        subreddit="test",
                        title="Test Post",
                        body="Test body",
                        author="testuser",
                        score=100,
                        upvote_ratio=0.95,
                        num_comments=10,
                        created_utc=1234567890.0,
                        url="https://reddit.com/r/test/comments/abc123/test_post/",
                        permalink="/r/test/comments/abc123/test_post/",
                        is_self=True,
                        over_18=False,
                    ),
                ),
            ],
            test_mock={
                "get_post": lambda creds, post_id: RedditPostDetails(
                    id="abc123",
                    subreddit="test",
                    title="Test Post",
                    body="Test body",
                    author="testuser",
                    score=100,
                    upvote_ratio=0.95,
                    num_comments=10,
                    created_utc=1234567890.0,
                    url="https://reddit.com/r/test/comments/abc123/test_post/",
                    permalink="/r/test/comments/abc123/test_post/",
                    is_self=True,
                    over_18=False,
                )
            },
        )

    @staticmethod
    def get_post(creds: RedditCredentials, post_id: str) -> RedditPostDetails:
        client = get_praw(creds)
        # Handle both 'abc123' and 't3_abc123' formats
        if post_id.startswith("t3_"):
            post_id = post_id[3:]
        submission = client.submission(id=post_id)

        return RedditPostDetails(
            id=submission.id,
            subreddit=submission.subreddit.display_name,
            title=submission.title,
            body=submission.selftext,
            author=str(submission.author) if submission.author else "[deleted]",
            score=submission.score,
            upvote_ratio=submission.upvote_ratio,
            num_comments=submission.num_comments,
            created_utc=submission.created_utc,
            url=submission.url,
            permalink=submission.permalink,
            is_self=submission.is_self,
            over_18=submission.over_18,
        )

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            post = self.get_post(credentials, input_data.post_id)
            yield "post", post
        except Exception as e:
            yield "error", str(e)


class GetUserPostsBlock(Block):
    """Get posts by a specific Reddit user."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        username: str = SchemaField(
            description="Reddit username to fetch posts from (without /u/ prefix)",
        )
        post_limit: int = SchemaField(
            description="Maximum number of posts to fetch",
            default=10,
        )
        sort: str = SchemaField(
            description="Sort order: 'new', 'hot', 'top', or 'controversial'",
            default="new",
        )

    class Output(BlockSchemaOutput):
        post: RedditPost = SchemaField(description="A post by the user")
        posts: list[RedditPost] = SchemaField(description="All posts by the user")
        error: str = SchemaField(
            description="Error message if posts couldn't be fetched"
        )

    def __init__(self):
        super().__init__(
            id="6fbe6329-d13e-4d2e-bd4d-b4d921b56161",
            description="Fetch posts by a specific Reddit user.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetUserPostsBlock.Input,
            output_schema=GetUserPostsBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "username": "testuser",
                "post_limit": 2,
            },
            test_output=[
                (
                    "post",
                    RedditPost(
                        id="id1", subreddit="sub1", title="title1", body="body1"
                    ),
                ),
                (
                    "post",
                    RedditPost(
                        id="id2", subreddit="sub2", title="title2", body="body2"
                    ),
                ),
                (
                    "posts",
                    [
                        RedditPost(
                            id="id1", subreddit="sub1", title="title1", body="body1"
                        ),
                        RedditPost(
                            id="id2", subreddit="sub2", title="title2", body="body2"
                        ),
                    ],
                ),
            ],
            test_mock={
                "get_user_posts": lambda creds, username, limit, sort: [
                    MockObject(
                        id="id1",
                        subreddit=MockObject(display_name="sub1"),
                        title="title1",
                        selftext="body1",
                    ),
                    MockObject(
                        id="id2",
                        subreddit=MockObject(display_name="sub2"),
                        title="title2",
                        selftext="body2",
                    ),
                ]
            },
        )

    @staticmethod
    def get_user_posts(
        creds: RedditCredentials, username: str, limit: int, sort: str
    ) -> list:
        client = get_praw(creds)
        redditor = client.redditor(username)

        if sort == "new":
            submissions = redditor.submissions.new(limit=limit)
        elif sort == "hot":
            submissions = redditor.submissions.hot(limit=limit)
        elif sort == "top":
            submissions = redditor.submissions.top(limit=limit)
        elif sort == "controversial":
            submissions = redditor.submissions.controversial(limit=limit)
        else:
            submissions = redditor.submissions.new(limit=limit)

        return list(submissions)

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            submissions = self.get_user_posts(
                credentials,
                input_data.username,
                input_data.post_limit,
                input_data.sort,
            )
            all_posts = []
            for submission in submissions:
                post = RedditPost(
                    id=submission.id,
                    subreddit=submission.subreddit.display_name,
                    title=submission.title,
                    body=submission.selftext,
                )
                all_posts.append(post)
                yield "post", post
            yield "posts", all_posts
        except Exception as e:
            yield "error", str(e)


class RedditSearchResult(BaseModel):
    """A search result from Reddit."""

    id: str
    subreddit: str
    title: str
    body: str
    author: str
    score: int
    num_comments: int
    created_utc: float
    permalink: str


class SearchRedditBlock(Block):
    """Search Reddit for posts matching a query."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        query: str = SchemaField(
            description="Search query string",
        )
        subreddit: str | None = SchemaField(
            description="Limit search to a specific subreddit (without /r/ prefix)",
            default=None,
        )
        sort: str = SchemaField(
            description="Sort order: 'relevance', 'hot', 'top', 'new', or 'comments'",
            default="relevance",
        )
        time_filter: str = SchemaField(
            description="Time filter: 'all', 'day', 'hour', 'month', 'week', or 'year'",
            default="all",
        )
        limit: int = SchemaField(
            description="Maximum number of results to return",
            default=10,
        )

    class Output(BlockSchemaOutput):
        result: RedditSearchResult = SchemaField(description="A search result")
        results: list[RedditSearchResult] = SchemaField(
            description="All search results"
        )
        error: str = SchemaField(description="Error message if search failed")

    def __init__(self):
        super().__init__(
            id="4a0c975e-807b-4d5e-83c9-1619864a4b1a",
            description="Search Reddit for posts matching a query. Can search all of Reddit or a specific subreddit.",
            categories={BlockCategory.SOCIAL},
            input_schema=SearchRedditBlock.Input,
            output_schema=SearchRedditBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "test query",
                "limit": 2,
            },
            test_output=[
                (
                    "result",
                    RedditSearchResult(
                        id="id1",
                        subreddit="sub1",
                        title="title1",
                        body="body1",
                        author="author1",
                        score=100,
                        num_comments=10,
                        created_utc=1234567890.0,
                        permalink="/r/sub1/comments/id1/title1/",
                    ),
                ),
                (
                    "result",
                    RedditSearchResult(
                        id="id2",
                        subreddit="sub2",
                        title="title2",
                        body="body2",
                        author="author2",
                        score=50,
                        num_comments=5,
                        created_utc=1234567891.0,
                        permalink="/r/sub2/comments/id2/title2/",
                    ),
                ),
                (
                    "results",
                    [
                        RedditSearchResult(
                            id="id1",
                            subreddit="sub1",
                            title="title1",
                            body="body1",
                            author="author1",
                            score=100,
                            num_comments=10,
                            created_utc=1234567890.0,
                            permalink="/r/sub1/comments/id1/title1/",
                        ),
                        RedditSearchResult(
                            id="id2",
                            subreddit="sub2",
                            title="title2",
                            body="body2",
                            author="author2",
                            score=50,
                            num_comments=5,
                            created_utc=1234567891.0,
                            permalink="/r/sub2/comments/id2/title2/",
                        ),
                    ],
                ),
            ],
            test_mock={
                "search_reddit": lambda creds, query, subreddit, sort, time_filter, limit: [
                    MockObject(
                        id="id1",
                        subreddit=MockObject(display_name="sub1"),
                        title="title1",
                        selftext="body1",
                        author="author1",
                        score=100,
                        num_comments=10,
                        created_utc=1234567890.0,
                        permalink="/r/sub1/comments/id1/title1/",
                    ),
                    MockObject(
                        id="id2",
                        subreddit=MockObject(display_name="sub2"),
                        title="title2",
                        selftext="body2",
                        author="author2",
                        score=50,
                        num_comments=5,
                        created_utc=1234567891.0,
                        permalink="/r/sub2/comments/id2/title2/",
                    ),
                ]
            },
        )

    @staticmethod
    def search_reddit(
        creds: RedditCredentials,
        query: str,
        subreddit: str | None,
        sort: str,
        time_filter: str,
        limit: int,
    ) -> list:
        client = get_praw(creds)

        if subreddit:
            sub = client.subreddit(subreddit)
            results = sub.search(query, sort=sort, time_filter=time_filter, limit=limit)
        else:
            results = client.subreddit("all").search(
                query, sort=sort, time_filter=time_filter, limit=limit
            )

        return list(results)

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            submissions = self.search_reddit(
                credentials,
                input_data.query,
                input_data.subreddit,
                input_data.sort,
                input_data.time_filter,
                input_data.limit,
            )
            all_results = []
            for submission in submissions:
                result = RedditSearchResult(
                    id=submission.id,
                    subreddit=submission.subreddit.display_name,
                    title=submission.title,
                    body=submission.selftext,
                    author=str(submission.author) if submission.author else "[deleted]",
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=submission.created_utc,
                    permalink=submission.permalink,
                )
                all_results.append(result)
                yield "result", result
            yield "results", all_results
        except Exception as e:
            yield "error", str(e)


class EditRedditPostBlock(Block):
    """Edit an existing Reddit post that you own."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        post_id: str = SchemaField(
            description="The ID of the post to edit (must be your own post)",
        )
        new_content: str = SchemaField(
            description="The new body text for the post",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the edit was successful")
        post_url: str = SchemaField(description="URL of the edited post")
        error: str = SchemaField(description="Error message if the edit failed")

    def __init__(self):
        super().__init__(
            id="cdb9df0f-8b1d-433e-873a-ededc1b6479d",
            description="Edit the body text of an existing Reddit post that you own. Only works for self/text posts.",
            categories={BlockCategory.SOCIAL},
            input_schema=EditRedditPostBlock.Input,
            output_schema=EditRedditPostBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "post_id": "abc123",
                "new_content": "Updated post content",
            },
            test_output=[
                ("success", True),
                ("post_url", "https://reddit.com/r/test/comments/abc123/test_post/"),
            ],
            test_mock={
                "edit_post": lambda creds, post_id, new_content: (
                    True,
                    "https://reddit.com/r/test/comments/abc123/test_post/",
                )
            },
        )

    @staticmethod
    def edit_post(
        creds: RedditCredentials, post_id: str, new_content: str
    ) -> tuple[bool, str]:
        client = get_praw(creds)
        # Handle both 'abc123' and 't3_abc123' formats
        if post_id.startswith("t3_"):
            post_id = post_id[3:]
        submission = client.submission(id=post_id)
        submission.edit(new_content)
        return True, f"https://reddit.com{submission.permalink}"

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            success, post_url = self.edit_post(
                credentials, input_data.post_id, input_data.new_content
            )
            yield "success", success
            yield "post_url", post_url
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg:
                error_msg = (
                    "Permission denied (403): You can only edit your own posts. "
                    "Make sure the post belongs to the authenticated Reddit account."
                )
            yield "error", error_msg


class SubredditInfo(BaseModel):
    """Information about a subreddit."""

    name: str
    display_name: str
    title: str
    description: str
    public_description: str
    subscribers: int
    created_utc: float
    over_18: bool
    url: str


class GetSubredditInfoBlock(Block):
    """Get information about a subreddit."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        subreddit: str = SchemaField(
            description="Subreddit name (without /r/ prefix)",
        )

    class Output(BlockSchemaOutput):
        info: SubredditInfo = SchemaField(description="Subreddit information")
        error: str = SchemaField(
            description="Error message if the subreddit couldn't be fetched"
        )

    def __init__(self):
        super().__init__(
            id="5a2d1f0c-01fb-43ea-bad7-2260d269c930",
            description="Get information about a subreddit including subscriber count, description, and rules.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetSubredditInfoBlock.Input,
            output_schema=GetSubredditInfoBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subreddit": "python",
            },
            test_output=[
                (
                    "info",
                    SubredditInfo(
                        name="t5_2qh0y",
                        display_name="python",
                        title="Python",
                        description="News about the Python programming language",
                        public_description="News about Python",
                        subscribers=1000000,
                        created_utc=1234567890.0,
                        over_18=False,
                        url="/r/python/",
                    ),
                ),
            ],
            test_mock={
                "get_subreddit_info": lambda creds, subreddit: SubredditInfo(
                    name="t5_2qh0y",
                    display_name="python",
                    title="Python",
                    description="News about the Python programming language",
                    public_description="News about Python",
                    subscribers=1000000,
                    created_utc=1234567890.0,
                    over_18=False,
                    url="/r/python/",
                )
            },
        )

    @staticmethod
    def get_subreddit_info(creds: RedditCredentials, subreddit: str) -> SubredditInfo:
        client = get_praw(creds)
        sub = client.subreddit(subreddit)

        return SubredditInfo(
            name=sub.name,
            display_name=sub.display_name,
            title=sub.title,
            description=sub.description,
            public_description=sub.public_description,
            subscribers=sub.subscribers,
            created_utc=sub.created_utc,
            over_18=sub.over18,
            url=sub.url,
        )

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            info = self.get_subreddit_info(credentials, input_data.subreddit)
            yield "info", info
        except Exception as e:
            yield "error", str(e)


class RedditCommentData(BaseModel):
    """Data about a Reddit comment."""

    id: str
    post_id: str
    parent_id: str
    author: str
    body: str
    score: int
    created_utc: float
    edited: bool
    is_submitter: bool
    permalink: str
    depth: int


class GetPostCommentsBlock(Block):
    """Get comments on a Reddit post."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        post_id: str = SchemaField(
            description="The ID of the post to get comments from",
        )
        limit: int = SchemaField(
            description="Maximum number of top-level comments to fetch (max 100)",
            default=25,
        )
        sort: str = SchemaField(
            description="Sort order: 'best', 'top', 'new', 'controversial', 'old', 'q&a'",
            default="best",
        )

    class Output(BlockSchemaOutput):
        comment: RedditCommentData = SchemaField(description="A comment on the post")
        comments: list[RedditCommentData] = SchemaField(
            description="All fetched comments"
        )
        error: str = SchemaField(description="Error message if comments couldn't be fetched")

    def __init__(self):
        super().__init__(
            id="98422b2c-c3b0-4d70-871f-56bd966f46da",
            description="Get top-level comments on a Reddit post.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetPostCommentsBlock.Input,
            output_schema=GetPostCommentsBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "post_id": "abc123",
                "limit": 2,
            },
            test_output=[
                (
                    "comment",
                    RedditCommentData(
                        id="comment1",
                        post_id="abc123",
                        parent_id="t3_abc123",
                        author="user1",
                        body="Comment body 1",
                        score=10,
                        created_utc=1234567890.0,
                        edited=False,
                        is_submitter=False,
                        permalink="/r/test/comments/abc123/test/comment1/",
                        depth=0,
                    ),
                ),
                (
                    "comment",
                    RedditCommentData(
                        id="comment2",
                        post_id="abc123",
                        parent_id="t3_abc123",
                        author="user2",
                        body="Comment body 2",
                        score=5,
                        created_utc=1234567891.0,
                        edited=False,
                        is_submitter=True,
                        permalink="/r/test/comments/abc123/test/comment2/",
                        depth=0,
                    ),
                ),
                (
                    "comments",
                    [
                        RedditCommentData(
                            id="comment1",
                            post_id="abc123",
                            parent_id="t3_abc123",
                            author="user1",
                            body="Comment body 1",
                            score=10,
                            created_utc=1234567890.0,
                            edited=False,
                            is_submitter=False,
                            permalink="/r/test/comments/abc123/test/comment1/",
                            depth=0,
                        ),
                        RedditCommentData(
                            id="comment2",
                            post_id="abc123",
                            parent_id="t3_abc123",
                            author="user2",
                            body="Comment body 2",
                            score=5,
                            created_utc=1234567891.0,
                            edited=False,
                            is_submitter=True,
                            permalink="/r/test/comments/abc123/test/comment2/",
                            depth=0,
                        ),
                    ],
                ),
            ],
            test_mock={
                "get_comments": lambda creds, post_id, limit, sort: [
                    MockObject(
                        id="comment1",
                        link_id="t3_abc123",
                        parent_id="t3_abc123",
                        author="user1",
                        body="Comment body 1",
                        score=10,
                        created_utc=1234567890.0,
                        edited=False,
                        is_submitter=False,
                        permalink="/r/test/comments/abc123/test/comment1/",
                        depth=0,
                    ),
                    MockObject(
                        id="comment2",
                        link_id="t3_abc123",
                        parent_id="t3_abc123",
                        author="user2",
                        body="Comment body 2",
                        score=5,
                        created_utc=1234567891.0,
                        edited=False,
                        is_submitter=True,
                        permalink="/r/test/comments/abc123/test/comment2/",
                        depth=0,
                    ),
                ]
            },
        )

    @staticmethod
    def get_comments(
        creds: RedditCredentials, post_id: str, limit: int, sort: str
    ) -> list:
        client = get_praw(creds)
        if post_id.startswith("t3_"):
            post_id = post_id[3:]
        submission = client.submission(id=post_id)
        submission.comment_sort = sort
        # Replace MoreComments with actual comments up to limit
        submission.comments.replace_more(limit=0)
        # Return only top-level comments (depth=0), limited
        # CommentForest supports indexing, so use slicing directly
        max_comments = min(limit, 100)
        return [submission.comments[i] for i in range(min(len(submission.comments), max_comments))]

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            comments = self.get_comments(
                credentials,
                input_data.post_id,
                input_data.limit,
                input_data.sort,
            )
            all_comments = []
            for comment in comments:
                # Extract post_id from link_id (format: t3_xxxxx)
                post_id = comment.link_id
                if post_id.startswith("t3_"):
                    post_id = post_id[3:]

                comment_data = RedditCommentData(
                    id=comment.id,
                    post_id=post_id,
                    parent_id=comment.parent_id,
                    author=str(comment.author) if comment.author else "[deleted]",
                    body=comment.body,
                    score=comment.score,
                    created_utc=comment.created_utc,
                    edited=bool(comment.edited),
                    is_submitter=comment.is_submitter,
                    permalink=comment.permalink,
                    depth=comment.depth,
                )
                all_comments.append(comment_data)
                yield "comment", comment_data
            yield "comments", all_comments
        except Exception as e:
            yield "error", str(e)


class GetCommentRepliesBlock(Block):
    """Get replies to a specific Reddit comment."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        comment_id: str = SchemaField(
            description="The ID of the comment to get replies from",
        )
        post_id: str = SchemaField(
            description="The ID of the post containing the comment",
        )
        limit: int = SchemaField(
            description="Maximum number of replies to fetch (max 50)",
            default=10,
        )

    class Output(BlockSchemaOutput):
        reply: RedditCommentData = SchemaField(description="A reply to the comment")
        replies: list[RedditCommentData] = SchemaField(description="All replies")
        error: str = SchemaField(description="Error message if replies couldn't be fetched")

    def __init__(self):
        super().__init__(
            id="7fa83965-7289-432f-98a9-1575f5bcc8f1",
            description="Get replies to a specific Reddit comment.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetCommentRepliesBlock.Input,
            output_schema=GetCommentRepliesBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "comment_id": "comment1",
                "post_id": "abc123",
                "limit": 2,
            },
            test_output=[
                (
                    "reply",
                    RedditCommentData(
                        id="reply1",
                        post_id="abc123",
                        parent_id="t1_comment1",
                        author="replier1",
                        body="Reply body 1",
                        score=3,
                        created_utc=1234567892.0,
                        edited=False,
                        is_submitter=False,
                        permalink="/r/test/comments/abc123/test/reply1/",
                        depth=1,
                    ),
                ),
                (
                    "replies",
                    [
                        RedditCommentData(
                            id="reply1",
                            post_id="abc123",
                            parent_id="t1_comment1",
                            author="replier1",
                            body="Reply body 1",
                            score=3,
                            created_utc=1234567892.0,
                            edited=False,
                            is_submitter=False,
                            permalink="/r/test/comments/abc123/test/reply1/",
                            depth=1,
                        ),
                    ],
                ),
            ],
            test_mock={
                "get_replies": lambda creds, comment_id, post_id, limit: [
                    MockObject(
                        id="reply1",
                        link_id="t3_abc123",
                        parent_id="t1_comment1",
                        author="replier1",
                        body="Reply body 1",
                        score=3,
                        created_utc=1234567892.0,
                        edited=False,
                        is_submitter=False,
                        permalink="/r/test/comments/abc123/test/reply1/",
                        depth=1,
                    ),
                ]
            },
        )

    @staticmethod
    def get_replies(
        creds: RedditCredentials, comment_id: str, post_id: str, limit: int
    ) -> list:
        client = get_praw(creds)
        if post_id.startswith("t3_"):
            post_id = post_id[3:]
        if comment_id.startswith("t1_"):
            comment_id = comment_id[3:]

        # Get the submission and find the comment
        submission = client.submission(id=post_id)
        submission.comments.replace_more(limit=0)

        # Find the target comment - filter out MoreComments which don't have .id
        comment = None
        for c in submission.comments.list():
            if isinstance(c, MoreComments):
                continue
            if c.id == comment_id:
                comment = c
                break

        if not comment:
            return []

        # Get direct replies - filter out MoreComments objects
        replies = []
        # CommentForest supports indexing
        for i in range(len(comment.replies)):
            reply = comment.replies[i]
            if isinstance(reply, MoreComments):
                continue
            replies.append(reply)
            if len(replies) >= min(limit, 50):
                break

        return replies

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            replies = self.get_replies(
                credentials,
                input_data.comment_id,
                input_data.post_id,
                input_data.limit,
            )
            all_replies = []
            for reply in replies:
                post_id = reply.link_id
                if post_id.startswith("t3_"):
                    post_id = post_id[3:]

                reply_data = RedditCommentData(
                    id=reply.id,
                    post_id=post_id,
                    parent_id=reply.parent_id,
                    author=str(reply.author) if reply.author else "[deleted]",
                    body=reply.body,
                    score=reply.score,
                    created_utc=reply.created_utc,
                    edited=bool(reply.edited),
                    is_submitter=reply.is_submitter,
                    permalink=reply.permalink,
                    depth=reply.depth,
                )
                all_replies.append(reply_data)
                yield "reply", reply_data
            yield "replies", all_replies
        except Exception as e:
            yield "error", str(e)


class GetCommentBlock(Block):
    """Get details about a specific Reddit comment."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        comment_id: str = SchemaField(
            description="The ID of the comment to fetch",
        )

    class Output(BlockSchemaOutput):
        comment: RedditCommentData = SchemaField(description="The comment details")
        error: str = SchemaField(description="Error message if comment couldn't be fetched")

    def __init__(self):
        super().__init__(
            id="72cb311a-5998-4e0a-9bc4-f1b67a97284e",
            description="Get details about a specific Reddit comment by its ID.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetCommentBlock.Input,
            output_schema=GetCommentBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "comment_id": "comment1",
            },
            test_output=[
                (
                    "comment",
                    RedditCommentData(
                        id="comment1",
                        post_id="abc123",
                        parent_id="t3_abc123",
                        author="user1",
                        body="Comment body",
                        score=10,
                        created_utc=1234567890.0,
                        edited=False,
                        is_submitter=False,
                        permalink="/r/test/comments/abc123/test/comment1/",
                        depth=0,
                    ),
                ),
            ],
            test_mock={
                "get_comment": lambda creds, comment_id: MockObject(
                    id="comment1",
                    link_id="t3_abc123",
                    parent_id="t3_abc123",
                    author="user1",
                    body="Comment body",
                    score=10,
                    created_utc=1234567890.0,
                    edited=False,
                    is_submitter=False,
                    permalink="/r/test/comments/abc123/test/comment1/",
                    depth=0,
                )
            },
        )

    @staticmethod
    def get_comment(creds: RedditCredentials, comment_id: str):
        client = get_praw(creds)
        if comment_id.startswith("t1_"):
            comment_id = comment_id[3:]
        return client.comment(id=comment_id)

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            comment = self.get_comment(credentials, input_data.comment_id)

            post_id = comment.link_id
            if post_id.startswith("t3_"):
                post_id = post_id[3:]

            comment_data = RedditCommentData(
                id=comment.id,
                post_id=post_id,
                parent_id=comment.parent_id,
                author=str(comment.author) if comment.author else "[deleted]",
                body=comment.body,
                score=comment.score,
                created_utc=comment.created_utc,
                edited=bool(comment.edited),
                is_submitter=comment.is_submitter,
                permalink=comment.permalink,
                depth=comment.depth,
            )
            yield "comment", comment_data
        except Exception as e:
            yield "error", str(e)
