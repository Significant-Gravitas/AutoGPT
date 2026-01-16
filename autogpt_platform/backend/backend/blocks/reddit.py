import logging
from datetime import datetime, timezone
from typing import Iterator, Literal

import praw
from praw.models import Comment, MoreComments, Submission
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

# Type aliases for Reddit API options
UserPostSort = Literal["new", "hot", "top", "controversial"]
SearchSort = Literal["relevance", "hot", "top", "new", "comments"]
TimeFilter = Literal["all", "day", "hour", "month", "week", "year"]
CommentSort = Literal["best", "top", "new", "controversial", "old", "qa"]
InboxType = Literal["all", "unread", "messages", "mentions", "comment_replies"]

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
    scopes=[
        "identity",
        "read",
        "submit",
        "edit",
        "history",
        "privatemessages",
        "flair",
    ],
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
    post_id: str
    subreddit: str
    title: str
    body: str


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


def strip_reddit_prefix(id_str: str) -> str:
    """
    Strip Reddit type prefix (t1_, t3_, etc.) from an ID if present.

    Reddit uses type prefixes like t1_ for comments, t3_ for posts, etc.
    This helper normalizes IDs by removing these prefixes when present,
    allowing blocks to accept both 'abc123' and 't3_abc123' formats.

    Args:
        id_str: The ID string that may have a Reddit type prefix.

    Returns:
        The ID without the type prefix.
    """
    if (
        len(id_str) > 3
        and id_str[0] == "t"
        and id_str[1].isdigit()
        and id_str[2] == "_"
    ):
        return id_str[3:]
    return id_str


class GetRedditPostsBlock(Block):
    class Input(BlockSchemaInput):
        subreddit: str = SchemaField(
            description="Subreddit name, excluding the /r/ prefix",
            default="writingprompts",
            advanced=False,
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
                        post_id="id1",
                        subreddit="subreddit",
                        title="title1",
                        body="body1",
                    ),
                ),
                (
                    "post",
                    RedditPost(
                        post_id="id2",
                        subreddit="subreddit",
                        title="title2",
                        body="body2",
                    ),
                ),
                (
                    "posts",
                    [
                        RedditPost(
                            post_id="id1",
                            subreddit="subreddit",
                            title="title1",
                            body="body1",
                        ),
                        RedditPost(
                            post_id="id2",
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
                    # Posts are ordered newest-first, so all subsequent posts will also be older
                    break

            if input_data.last_post and post.id == input_data.last_post:
                break

            reddit_post = RedditPost(
                post_id=post.id,
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
        post_id: str = SchemaField(
            description="The post ID (pass-through for chaining)"
        )

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
                "post_id": "test_post_id",
                "comment": "comment",
            },
            test_output=[
                ("comment_id", "dummy_comment_id"),
                ("post_id", "test_post_id"),
            ],
            test_mock={
                "reply_post": lambda creds, post_id, comment: "dummy_comment_id"
            },
        )

    @staticmethod
    def reply_post(creds: RedditCredentials, post_id: str, comment: str) -> str:
        client = get_praw(creds)
        post_id = strip_reddit_prefix(post_id)
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
        yield "post_id", input_data.post_id


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
            description="Flair template ID to apply to the post (from GetSubredditFlairsBlock)",
            default=None,
        )
        flair_text: str | None = SchemaField(
            description="Custom flair text (only used if the flair template allows editing)",
            default=None,
        )

    class Output(BlockSchemaOutput):
        post_id: str = SchemaField(description="ID of the created post")
        post_url: str = SchemaField(description="URL of the created post")
        subreddit: str = SchemaField(
            description="The subreddit name (pass-through for chaining)"
        )

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
                ("subreddit", "test"),
            ],
            test_mock={
                "create_post": lambda creds, subreddit, title, content, url, flair_id, flair_text: (
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
        flair_text: str | None = None,
    ) -> tuple[str, str]:
        """
        Create a new post on a subreddit.

        Args:
            creds: Reddit OAuth2 credentials
            subreddit: Subreddit name (without /r/ prefix)
            title: Post title
            content: Post body text (for text posts)
            url: URL to submit (for link posts, overrides content)
            flair_id: Optional flair template ID to apply
            flair_text: Optional custom flair text (for editable flairs)

        Returns:
            Tuple of (post_id, post_url)
        """
        client = get_praw(creds)
        sub = client.subreddit(subreddit)

        if url:
            submission = sub.submit(
                title=title, url=url, flair_id=flair_id, flair_text=flair_text
            )
        else:
            submission = sub.submit(
                title=title, selftext=content, flair_id=flair_id, flair_text=flair_text
            )

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
            input_data.flair_text,
        )
        yield "post_id", post_id
        yield "post_url", post_url
        yield "subreddit", input_data.subreddit


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
        post_id = strip_reddit_prefix(post_id)
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
        sort: UserPostSort = SchemaField(
            description="Sort order for user posts",
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
                        post_id="id1", subreddit="sub1", title="title1", body="body1"
                    ),
                ),
                (
                    "post",
                    RedditPost(
                        post_id="id2", subreddit="sub2", title="title2", body="body2"
                    ),
                ),
                (
                    "posts",
                    [
                        RedditPost(
                            post_id="id1",
                            subreddit="sub1",
                            title="title1",
                            body="body1",
                        ),
                        RedditPost(
                            post_id="id2",
                            subreddit="sub2",
                            title="title2",
                            body="body2",
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
        creds: RedditCredentials, username: str, limit: int, sort: UserPostSort
    ) -> list[Submission]:
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
                    post_id=submission.id,
                    subreddit=submission.subreddit.display_name,
                    title=submission.title,
                    body=submission.selftext,
                )
                all_posts.append(post)
                yield "post", post
            yield "posts", all_posts
        except Exception as e:
            yield "error", str(e)


class RedditGetMyPostsBlock(Block):
    """Get posts by the authenticated Reddit user."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        post_limit: int = SchemaField(
            description="Maximum number of posts to fetch",
            default=10,
        )
        sort: UserPostSort = SchemaField(
            description="Sort order for posts",
            default="new",
        )

    class Output(BlockSchemaOutput):
        post: RedditPost = SchemaField(description="A post by you")
        posts: list[RedditPost] = SchemaField(description="All your posts")
        error: str = SchemaField(
            description="Error message if posts couldn't be fetched"
        )

    def __init__(self):
        super().__init__(
            id="4ab3381b-0c07-4201-89b3-fa2ec264f154",
            description="Fetch posts created by the authenticated Reddit user (you).",
            categories={BlockCategory.SOCIAL},
            input_schema=RedditGetMyPostsBlock.Input,
            output_schema=RedditGetMyPostsBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "post_limit": 2,
            },
            test_output=[
                (
                    "post",
                    RedditPost(
                        post_id="id1", subreddit="sub1", title="title1", body="body1"
                    ),
                ),
                (
                    "post",
                    RedditPost(
                        post_id="id2", subreddit="sub2", title="title2", body="body2"
                    ),
                ),
                (
                    "posts",
                    [
                        RedditPost(
                            post_id="id1",
                            subreddit="sub1",
                            title="title1",
                            body="body1",
                        ),
                        RedditPost(
                            post_id="id2",
                            subreddit="sub2",
                            title="title2",
                            body="body2",
                        ),
                    ],
                ),
            ],
            test_mock={
                "get_my_posts": lambda creds, limit, sort: [
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
    def get_my_posts(
        creds: RedditCredentials, limit: int, sort: UserPostSort
    ) -> list[Submission]:
        client = get_praw(creds)
        me = client.user.me()
        if not me:
            raise ValueError("Could not get authenticated user.")

        if sort == "new":
            submissions = me.submissions.new(limit=limit)
        elif sort == "hot":
            submissions = me.submissions.hot(limit=limit)
        elif sort == "top":
            submissions = me.submissions.top(limit=limit)
        elif sort == "controversial":
            submissions = me.submissions.controversial(limit=limit)
        else:
            submissions = me.submissions.new(limit=limit)

        return list(submissions)

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            submissions = self.get_my_posts(
                credentials,
                input_data.post_limit,
                input_data.sort,
            )
            all_posts = []
            for submission in submissions:
                post = RedditPost(
                    post_id=submission.id,
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
        sort: SearchSort = SchemaField(
            description="Sort order for search results",
            default="relevance",
        )
        time_filter: TimeFilter = SchemaField(
            description="Time filter for search results",
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
        sort: SearchSort,
        time_filter: TimeFilter,
        limit: int,
    ) -> list[Submission]:
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
        post_id: str = SchemaField(
            description="The post ID (pass-through for chaining)"
        )
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
                ("post_id", "abc123"),
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
        post_id = strip_reddit_prefix(post_id)
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
            yield "post_id", input_data.post_id
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
        subreddit: str = SchemaField(
            description="The subreddit name (pass-through for chaining)"
        )
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
                ("subreddit", "python"),
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
            yield "subreddit", input_data.subreddit
        except Exception as e:
            yield "error", str(e)


class RedditComment(BaseModel):
    """A Reddit comment."""

    comment_id: str
    post_id: str
    parent_comment_id: str | None
    author: str
    body: str
    score: int
    created_utc: float
    edited: bool
    is_submitter: bool
    permalink: str
    depth: int


class GetRedditPostCommentsBlock(Block):
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
        sort: CommentSort = SchemaField(
            description="Sort order for comments",
            default="best",
        )

    class Output(BlockSchemaOutput):
        comment: RedditComment = SchemaField(description="A comment on the post")
        comments: list[RedditComment] = SchemaField(description="All fetched comments")
        post_id: str = SchemaField(
            description="The post ID (pass-through for chaining)"
        )
        error: str = SchemaField(
            description="Error message if comments couldn't be fetched"
        )

    def __init__(self):
        super().__init__(
            id="98422b2c-c3b0-4d70-871f-56bd966f46da",
            description="Get top-level comments on a Reddit post.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetRedditPostCommentsBlock.Input,
            output_schema=GetRedditPostCommentsBlock.Output,
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
                    RedditComment(
                        comment_id="comment1",
                        post_id="abc123",
                        parent_comment_id=None,
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
                    RedditComment(
                        comment_id="comment2",
                        post_id="abc123",
                        parent_comment_id=None,
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
                        RedditComment(
                            comment_id="comment1",
                            post_id="abc123",
                            parent_comment_id=None,
                            author="user1",
                            body="Comment body 1",
                            score=10,
                            created_utc=1234567890.0,
                            edited=False,
                            is_submitter=False,
                            permalink="/r/test/comments/abc123/test/comment1/",
                            depth=0,
                        ),
                        RedditComment(
                            comment_id="comment2",
                            post_id="abc123",
                            parent_comment_id=None,
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
                ("post_id", "abc123"),
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
        creds: RedditCredentials, post_id: str, limit: int, sort: CommentSort
    ) -> list[Comment]:
        client = get_praw(creds)
        post_id = strip_reddit_prefix(post_id)
        submission = client.submission(id=post_id)
        submission.comment_sort = sort
        # Replace MoreComments with actual comments up to limit
        submission.comments.replace_more(limit=0)
        # Return only top-level comments (depth=0), limited
        # CommentForest supports indexing, so use slicing directly
        max_comments = min(limit, 100)
        return [
            submission.comments[i]
            for i in range(min(len(submission.comments), max_comments))
        ]

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
                comment_post_id = strip_reddit_prefix(comment.link_id)

                # parent_comment_id is None for top-level comments (parent is a post: t3_)
                # For replies, extract the comment ID from t1_xxxxx
                parent_comment_id = None
                if comment.parent_id.startswith("t1_"):
                    parent_comment_id = strip_reddit_prefix(comment.parent_id)

                comment_data = RedditComment(
                    comment_id=comment.id,
                    post_id=comment_post_id,
                    parent_comment_id=parent_comment_id,
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
            yield "post_id", input_data.post_id
        except Exception as e:
            yield "error", str(e)


class GetRedditCommentRepliesBlock(Block):
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
        reply: RedditComment = SchemaField(description="A reply to the comment")
        replies: list[RedditComment] = SchemaField(description="All replies")
        comment_id: str = SchemaField(
            description="The parent comment ID (pass-through for chaining)"
        )
        post_id: str = SchemaField(
            description="The post ID (pass-through for chaining)"
        )
        error: str = SchemaField(
            description="Error message if replies couldn't be fetched"
        )

    def __init__(self):
        super().__init__(
            id="7fa83965-7289-432f-98a9-1575f5bcc8f1",
            description="Get replies to a specific Reddit comment.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetRedditCommentRepliesBlock.Input,
            output_schema=GetRedditCommentRepliesBlock.Output,
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
                    RedditComment(
                        comment_id="reply1",
                        post_id="abc123",
                        parent_comment_id="comment1",
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
                        RedditComment(
                            comment_id="reply1",
                            post_id="abc123",
                            parent_comment_id="comment1",
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
                ("comment_id", "comment1"),
                ("post_id", "abc123"),
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
    ) -> list[Comment]:
        client = get_praw(creds)
        post_id = strip_reddit_prefix(post_id)
        comment_id = strip_reddit_prefix(comment_id)

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
                reply_post_id = strip_reddit_prefix(reply.link_id)

                # parent_comment_id is the parent comment (always present for replies)
                parent_comment_id = None
                if reply.parent_id.startswith("t1_"):
                    parent_comment_id = strip_reddit_prefix(reply.parent_id)

                reply_data = RedditComment(
                    comment_id=reply.id,
                    post_id=reply_post_id,
                    parent_comment_id=parent_comment_id,
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
            yield "comment_id", input_data.comment_id
            yield "post_id", input_data.post_id
        except Exception as e:
            yield "error", str(e)


class GetRedditCommentBlock(Block):
    """Get details about a specific Reddit comment."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        comment_id: str = SchemaField(
            description="The ID of the comment to fetch",
        )

    class Output(BlockSchemaOutput):
        comment: RedditComment = SchemaField(description="The comment details")
        error: str = SchemaField(
            description="Error message if comment couldn't be fetched"
        )

    def __init__(self):
        super().__init__(
            id="72cb311a-5998-4e0a-9bc4-f1b67a97284e",
            description="Get details about a specific Reddit comment by its ID.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetRedditCommentBlock.Input,
            output_schema=GetRedditCommentBlock.Output,
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
                    RedditComment(
                        comment_id="comment1",
                        post_id="abc123",
                        parent_comment_id=None,
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
        comment_id = strip_reddit_prefix(comment_id)
        return client.comment(id=comment_id)

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            comment = self.get_comment(credentials, input_data.comment_id)

            post_id = strip_reddit_prefix(comment.link_id)

            # parent_comment_id is None for top-level comments (parent is a post: t3_)
            parent_comment_id = None
            if comment.parent_id.startswith("t1_"):
                parent_comment_id = strip_reddit_prefix(comment.parent_id)

            comment_data = RedditComment(
                comment_id=comment.id,
                post_id=post_id,
                parent_comment_id=parent_comment_id,
                author=str(comment.author) if comment.author else "[deleted]",
                body=comment.body,
                score=comment.score,
                created_utc=comment.created_utc,
                edited=bool(comment.edited),
                is_submitter=comment.is_submitter,
                permalink=comment.permalink,
                # depth is only available when comments are fetched as part of a tree,
                # not when fetched directly by ID
                depth=getattr(comment, "depth", 0),
            )
            yield "comment", comment_data
        except Exception as e:
            yield "error", str(e)


class ReplyToRedditCommentBlock(Block):
    """Reply to a specific Reddit comment."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        comment_id: str = SchemaField(
            description="The ID of the comment to reply to",
        )
        reply_text: str = SchemaField(
            description="The text content of the reply",
        )

    class Output(BlockSchemaOutput):
        comment_id: str = SchemaField(description="ID of the newly created reply")
        parent_comment_id: str = SchemaField(
            description="The parent comment ID (pass-through for chaining)"
        )
        error: str = SchemaField(description="Error message if reply failed")

    def __init__(self):
        super().__init__(
            id="7635b059-3a9f-4f7d-b499-1b56c4f76f4f",
            description="Reply to a specific Reddit comment. Useful for threaded conversations.",
            categories={BlockCategory.SOCIAL},
            input_schema=ReplyToRedditCommentBlock.Input,
            output_schema=ReplyToRedditCommentBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "comment_id": "parent_comment",
                "reply_text": "This is a reply",
            },
            test_output=[
                ("comment_id", "new_reply_id"),
                ("parent_comment_id", "parent_comment"),
            ],
            test_mock={
                "reply_to_comment": lambda creds, comment_id, reply_text: "new_reply_id"
            },
        )

    @staticmethod
    def reply_to_comment(
        creds: RedditCredentials, comment_id: str, reply_text: str
    ) -> str:
        client = get_praw(creds)
        comment_id = strip_reddit_prefix(comment_id)
        comment = client.comment(id=comment_id)
        reply = comment.reply(reply_text)
        if not reply:
            raise ValueError("Failed to post reply.")
        return reply.id

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            new_comment_id = self.reply_to_comment(
                credentials, input_data.comment_id, input_data.reply_text
            )
            yield "comment_id", new_comment_id
            yield "parent_comment_id", input_data.comment_id
        except Exception as e:
            yield "error", str(e)


class RedditUserProfileSubreddit(BaseModel):
    """Information about a user's profile subreddit."""

    name: str
    title: str
    public_description: str
    subscribers: int
    over_18: bool


class RedditUserInfo(BaseModel):
    """Information about a Reddit user."""

    username: str
    user_id: str
    comment_karma: int
    link_karma: int
    total_karma: int
    created_utc: float
    is_gold: bool
    is_mod: bool
    has_verified_email: bool
    moderated_subreddits: list[str]
    profile_subreddit: RedditUserProfileSubreddit | None


class GetRedditUserInfoBlock(Block):
    """Get information about a Reddit user."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        username: str = SchemaField(
            description="The Reddit username to look up (without /u/ prefix)",
        )

    class Output(BlockSchemaOutput):
        user: RedditUserInfo = SchemaField(description="User information")
        username: str = SchemaField(
            description="The username (pass-through for chaining)"
        )
        error: str = SchemaField(description="Error message if user lookup failed")

    def __init__(self):
        super().__init__(
            id="1b4c6bd1-4f28-4bad-9ae9-e7034a0f61ff",
            description="Get information about a Reddit user including karma, account age, and verification status.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetRedditUserInfoBlock.Input,
            output_schema=GetRedditUserInfoBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "username": "testuser",
            },
            test_output=[
                (
                    "user",
                    RedditUserInfo(
                        username="testuser",
                        user_id="abc123",
                        comment_karma=1000,
                        link_karma=500,
                        total_karma=1500,
                        created_utc=1234567890.0,
                        is_gold=False,
                        is_mod=True,
                        has_verified_email=True,
                        moderated_subreddits=["python", "learnpython"],
                        profile_subreddit=RedditUserProfileSubreddit(
                            name="u_testuser",
                            title="testuser's profile",
                            public_description="A test user",
                            subscribers=100,
                            over_18=False,
                        ),
                    ),
                ),
                ("username", "testuser"),
            ],
            test_mock={
                "get_user_info": lambda creds, username: MockObject(
                    name="testuser",
                    id="abc123",
                    comment_karma=1000,
                    link_karma=500,
                    total_karma=1500,
                    created_utc=1234567890.0,
                    is_gold=False,
                    is_mod=True,
                    has_verified_email=True,
                    subreddit=MockObject(
                        display_name="u_testuser",
                        title="testuser's profile",
                        public_description="A test user",
                        subscribers=100,
                        over18=False,
                    ),
                ),
                "get_moderated_subreddits": lambda creds, username: [
                    MockObject(display_name="python"),
                    MockObject(display_name="learnpython"),
                ],
            },
        )

    @staticmethod
    def get_user_info(creds: RedditCredentials, username: str):
        client = get_praw(creds)
        if username.startswith("u/"):
            username = username[2:]
        return client.redditor(username)

    @staticmethod
    def get_moderated_subreddits(creds: RedditCredentials, username: str) -> list:
        client = get_praw(creds)
        if username.startswith("u/"):
            username = username[2:]
        redditor = client.redditor(username)
        return list(redditor.moderated())

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            redditor = self.get_user_info(credentials, input_data.username)
            moderated = self.get_moderated_subreddits(credentials, input_data.username)

            # Extract moderated subreddit names
            moderated_subreddits = [sub.display_name for sub in moderated]

            # Get profile subreddit info if available
            profile_subreddit = None
            if hasattr(redditor, "subreddit") and redditor.subreddit:
                try:
                    profile_subreddit = RedditUserProfileSubreddit(
                        name=redditor.subreddit.display_name,
                        title=redditor.subreddit.title or "",
                        public_description=redditor.subreddit.public_description or "",
                        subscribers=redditor.subreddit.subscribers or 0,
                        over_18=(
                            redditor.subreddit.over18
                            if hasattr(redditor.subreddit, "over18")
                            else False
                        ),
                    )
                except Exception:
                    # Profile subreddit may not be accessible
                    pass

            user_info = RedditUserInfo(
                username=redditor.name,
                user_id=redditor.id,
                comment_karma=redditor.comment_karma,
                link_karma=redditor.link_karma,
                total_karma=redditor.total_karma,
                created_utc=redditor.created_utc,
                is_gold=redditor.is_gold,
                is_mod=redditor.is_mod,
                has_verified_email=redditor.has_verified_email,
                moderated_subreddits=moderated_subreddits,
                profile_subreddit=profile_subreddit,
            )
            yield "user", user_info
            yield "username", input_data.username
        except Exception as e:
            yield "error", str(e)


class SendRedditMessageBlock(Block):
    """Send a private message to a Reddit user."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        username: str = SchemaField(
            description="The Reddit username to send a message to (without /u/ prefix)",
        )
        subject: str = SchemaField(
            description="The subject line of the message",
        )
        message: str = SchemaField(
            description="The body content of the message",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the message was sent")
        username: str = SchemaField(
            description="The username (pass-through for chaining)"
        )
        error: str = SchemaField(description="Error message if sending failed")

    def __init__(self):
        super().__init__(
            id="7921101a-0537-4259-82ea-bc186ca6b1b6",
            description="Send a private message (DM) to a Reddit user.",
            categories={BlockCategory.SOCIAL},
            input_schema=SendRedditMessageBlock.Input,
            output_schema=SendRedditMessageBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "username": "testuser",
                "subject": "Hello",
                "message": "This is a test message",
            },
            test_output=[
                ("success", True),
                ("username", "testuser"),
            ],
            test_mock={"send_message": lambda creds, username, subject, message: True},
        )

    @staticmethod
    def send_message(
        creds: RedditCredentials, username: str, subject: str, message: str
    ) -> bool:
        client = get_praw(creds)
        if username.startswith("u/"):
            username = username[2:]
        redditor = client.redditor(username)
        redditor.message(subject=subject, message=message)
        return True

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            success = self.send_message(
                credentials,
                input_data.username,
                input_data.subject,
                input_data.message,
            )
            yield "success", success
            yield "username", input_data.username
        except Exception as e:
            yield "error", str(e)


class RedditInboxItem(BaseModel):
    """A Reddit inbox item (message, comment reply, or mention)."""

    item_id: str
    item_type: str  # "message", "comment_reply", "mention"
    subject: str
    body: str
    author: str
    created_utc: float
    is_read: bool
    context: str | None  # permalink for comments, None for messages


class GetRedditInboxBlock(Block):
    """Get messages and notifications from Reddit inbox."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        inbox_type: InboxType = SchemaField(
            description="Type of inbox items to fetch",
            default="unread",
        )
        limit: int = SchemaField(
            description="Maximum number of items to fetch",
            default=25,
        )
        mark_read: bool = SchemaField(
            description="Whether to mark fetched items as read",
            default=False,
        )

    class Output(BlockSchemaOutput):
        item: RedditInboxItem = SchemaField(description="An inbox item")
        items: list[RedditInboxItem] = SchemaField(description="All fetched items")
        error: str = SchemaField(description="Error message if fetch failed")

    def __init__(self):
        super().__init__(
            id="5a91bb34-7ffe-4b9e-957b-9d4f8fe8dbc9",
            description="Get messages, mentions, and comment replies from your Reddit inbox.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetRedditInboxBlock.Input,
            output_schema=GetRedditInboxBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "inbox_type": "unread",
                "limit": 10,
            },
            test_output=[
                (
                    "item",
                    RedditInboxItem(
                        item_id="msg123",
                        item_type="message",
                        subject="Hello",
                        body="Test message body",
                        author="sender_user",
                        created_utc=1234567890.0,
                        is_read=False,
                        context=None,
                    ),
                ),
                (
                    "items",
                    [
                        RedditInboxItem(
                            item_id="msg123",
                            item_type="message",
                            subject="Hello",
                            body="Test message body",
                            author="sender_user",
                            created_utc=1234567890.0,
                            is_read=False,
                            context=None,
                        ),
                    ],
                ),
            ],
            test_mock={
                "get_inbox": lambda creds, inbox_type, limit: [
                    MockObject(
                        id="msg123",
                        subject="Hello",
                        body="Test message body",
                        author="sender_user",
                        created_utc=1234567890.0,
                        new=True,
                        context=None,
                        was_comment=False,
                    ),
                ]
            },
        )

    @staticmethod
    def get_inbox(creds: RedditCredentials, inbox_type: InboxType, limit: int) -> list:
        client = get_praw(creds)
        inbox = client.inbox

        if inbox_type == "all":
            items = inbox.all(limit=limit)
        elif inbox_type == "unread":
            items = inbox.unread(limit=limit)
        elif inbox_type == "messages":
            items = inbox.messages(limit=limit)
        elif inbox_type == "mentions":
            items = inbox.mentions(limit=limit)
        elif inbox_type == "comment_replies":
            items = inbox.comment_replies(limit=limit)
        else:
            items = inbox.unread(limit=limit)

        return list(items)

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            raw_items = self.get_inbox(
                credentials, input_data.inbox_type, input_data.limit
            )
            all_items = []

            for item in raw_items:
                # Determine item type
                if hasattr(item, "was_comment") and item.was_comment:
                    if hasattr(item, "subject") and "mention" in item.subject.lower():
                        item_type = "mention"
                    else:
                        item_type = "comment_reply"
                else:
                    item_type = "message"

                inbox_item = RedditInboxItem(
                    item_id=item.id,
                    item_type=item_type,
                    subject=item.subject if hasattr(item, "subject") else "",
                    body=item.body,
                    author=str(item.author) if item.author else "[deleted]",
                    created_utc=item.created_utc,
                    is_read=not item.new,
                    context=item.context if hasattr(item, "context") else None,
                )
                all_items.append(inbox_item)
                yield "item", inbox_item

            # Mark as read if requested
            if input_data.mark_read and raw_items:
                client = get_praw(credentials)
                client.inbox.mark_read(raw_items)

            yield "items", all_items
        except Exception as e:
            yield "error", str(e)


class DeleteRedditPostBlock(Block):
    """Delete a Reddit post that you own."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        post_id: str = SchemaField(
            description="The ID of the post to delete (must be your own post)",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the deletion was successful")
        post_id: str = SchemaField(
            description="The post ID (pass-through for chaining)"
        )
        error: str = SchemaField(description="Error message if deletion failed")

    def __init__(self):
        super().__init__(
            id="72e4730a-d66d-4785-8e54-5ab3af450c81",
            description="Delete a Reddit post that you own.",
            categories={BlockCategory.SOCIAL},
            input_schema=DeleteRedditPostBlock.Input,
            output_schema=DeleteRedditPostBlock.Output,
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
                ("success", True),
                ("post_id", "abc123"),
            ],
            test_mock={"delete_post": lambda creds, post_id: True},
        )

    @staticmethod
    def delete_post(creds: RedditCredentials, post_id: str) -> bool:
        client = get_praw(creds)
        post_id = strip_reddit_prefix(post_id)
        submission = client.submission(id=post_id)
        submission.delete()
        return True

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            success = self.delete_post(credentials, input_data.post_id)
            yield "success", success
            yield "post_id", input_data.post_id
        except Exception as e:
            yield "error", str(e)


class DeleteRedditCommentBlock(Block):
    """Delete a Reddit comment that you own."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        comment_id: str = SchemaField(
            description="The ID of the comment to delete (must be your own comment)",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the deletion was successful")
        comment_id: str = SchemaField(
            description="The comment ID (pass-through for chaining)"
        )
        error: str = SchemaField(description="Error message if deletion failed")

    def __init__(self):
        super().__init__(
            id="2650584d-434f-46db-81ef-26c8d8d41f81",
            description="Delete a Reddit comment that you own.",
            categories={BlockCategory.SOCIAL},
            input_schema=DeleteRedditCommentBlock.Input,
            output_schema=DeleteRedditCommentBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "comment_id": "xyz789",
            },
            test_output=[
                ("success", True),
                ("comment_id", "xyz789"),
            ],
            test_mock={"delete_comment": lambda creds, comment_id: True},
        )

    @staticmethod
    def delete_comment(creds: RedditCredentials, comment_id: str) -> bool:
        client = get_praw(creds)
        comment_id = strip_reddit_prefix(comment_id)
        comment = client.comment(id=comment_id)
        comment.delete()
        return True

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            success = self.delete_comment(credentials, input_data.comment_id)
            yield "success", success
            yield "comment_id", input_data.comment_id
        except Exception as e:
            yield "error", str(e)


class SubredditFlair(BaseModel):
    """A subreddit link flair template."""

    flair_id: str
    text: str
    text_editable: bool
    css_class: str = ""  # The CSS class for styling (from flair_css_class)


class GetSubredditFlairsBlock(Block):
    """Get available link flairs for a subreddit."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        subreddit: str = SchemaField(
            description="Subreddit name (without /r/ prefix)",
        )

    class Output(BlockSchemaOutput):
        flair: SubredditFlair = SchemaField(description="A flair option")
        flairs: list[SubredditFlair] = SchemaField(description="All available flairs")
        subreddit: str = SchemaField(
            description="The subreddit name (pass-through for chaining)"
        )
        error: str = SchemaField(description="Error message if fetch failed")

    def __init__(self):
        super().__init__(
            id="ada08f34-a7a9-44aa-869f-0638fa4e0a84",
            description="Get available link flair options for a subreddit.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetSubredditFlairsBlock.Input,
            output_schema=GetSubredditFlairsBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subreddit": "test",
            },
            test_output=[
                (
                    "flair",
                    SubredditFlair(
                        flair_id="abc123",
                        text="Discussion",
                        text_editable=False,
                        css_class="discussion",
                    ),
                ),
                (
                    "flairs",
                    [
                        SubredditFlair(
                            flair_id="abc123",
                            text="Discussion",
                            text_editable=False,
                            css_class="discussion",
                        ),
                    ],
                ),
                ("subreddit", "test"),
            ],
            test_mock={
                "get_flairs": lambda creds, subreddit: [
                    {
                        "flair_template_id": "abc123",
                        "flair_text": "Discussion",
                        "flair_text_editable": False,
                        "flair_css_class": "discussion",
                    },
                ]
            },
        )

    @staticmethod
    def get_flairs(creds: RedditCredentials, subreddit: str) -> list:
        client = get_praw(creds)
        # Use /r/{subreddit}/api/flairselector endpoint directly with is_newlink=True
        # This returns link flairs available for new submissions without requiring mod access
        # The link_templates API is moderator-only, so we use flairselector instead
        # Path must include the subreddit prefix per Reddit API docs
        response = client.post(
            f"r/{subreddit}/api/flairselector",
            data={"is_newlink": "true"},
        )
        # Response contains 'choices' list with available flairs
        choices = response.get("choices", [])
        return choices

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            raw_flairs = self.get_flairs(credentials, input_data.subreddit)
            all_flairs = []

            for flair in raw_flairs:
                # /api/flairselector returns flairs with flair_template_id, flair_text, etc.
                flair_data = SubredditFlair(
                    flair_id=flair.get("flair_template_id", ""),
                    text=flair.get("flair_text", ""),
                    text_editable=flair.get("flair_text_editable", False),
                    css_class=flair.get("flair_css_class", ""),
                )
                all_flairs.append(flair_data)
                yield "flair", flair_data

            yield "flairs", all_flairs
            yield "subreddit", input_data.subreddit
        except Exception as e:
            yield "error", str(e)


class SubredditRule(BaseModel):
    """A subreddit rule."""

    short_name: str
    description: str
    kind: str  # "all", "link", "comment"
    violation_reason: str
    priority: int


class GetSubredditRulesBlock(Block):
    """Get the rules for a subreddit."""

    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        subreddit: str = SchemaField(
            description="Subreddit name (without /r/ prefix)",
        )

    class Output(BlockSchemaOutput):
        rule: SubredditRule = SchemaField(description="A subreddit rule")
        rules: list[SubredditRule] = SchemaField(description="All subreddit rules")
        subreddit: str = SchemaField(
            description="The subreddit name (pass-through for chaining)"
        )
        error: str = SchemaField(description="Error message if fetch failed")

    def __init__(self):
        super().__init__(
            id="222aa36c-fa70-4879-8e8a-37d100175f5c",
            description="Get the rules for a subreddit to ensure compliance before posting.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetSubredditRulesBlock.Input,
            output_schema=GetSubredditRulesBlock.Output,
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subreddit": "test",
            },
            test_output=[
                (
                    "rule",
                    SubredditRule(
                        short_name="No spam",
                        description="Do not post spam or self-promotional content.",
                        kind="all",
                        violation_reason="Spam",
                        priority=0,
                    ),
                ),
                (
                    "rules",
                    [
                        SubredditRule(
                            short_name="No spam",
                            description="Do not post spam or self-promotional content.",
                            kind="all",
                            violation_reason="Spam",
                            priority=0,
                        ),
                    ],
                ),
                ("subreddit", "test"),
            ],
            test_mock={
                "get_rules": lambda creds, subreddit: [
                    MockObject(
                        short_name="No spam",
                        description="Do not post spam or self-promotional content.",
                        kind="all",
                        violation_reason="Spam",
                        priority=0,
                    ),
                ]
            },
        )

    @staticmethod
    def get_rules(creds: RedditCredentials, subreddit: str) -> list:
        client = get_praw(creds)
        sub = client.subreddit(subreddit)
        return list(sub.rules)

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        try:
            raw_rules = self.get_rules(credentials, input_data.subreddit)
            all_rules = []

            for idx, rule in enumerate(raw_rules):
                rule_data = SubredditRule(
                    short_name=rule.short_name,
                    description=rule.description or "",
                    kind=rule.kind,
                    violation_reason=rule.violation_reason or rule.short_name,
                    priority=idx,
                )
                all_rules.append(rule_data)
                yield "rule", rule_data

            yield "rules", all_rules
            yield "subreddit", input_data.subreddit
        except Exception as e:
            yield "error", str(e)
