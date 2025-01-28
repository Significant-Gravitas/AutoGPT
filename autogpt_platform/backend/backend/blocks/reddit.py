from datetime import datetime, timezone
from typing import Iterator, Literal

import praw
from pydantic import BaseModel, SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
    UserPasswordCredentials,
)
from backend.integrations.providers import ProviderName
from backend.util.mock import MockObject
from backend.util.settings import Settings

RedditCredentials = UserPasswordCredentials
RedditCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.REDDIT],
    Literal["user_password"],
]


def RedditCredentialsField() -> RedditCredentialsInput:
    """Creates a Reddit credentials input on a block."""
    return CredentialsField(
        description="The Reddit integration requires a username and password.",
    )


TEST_CREDENTIALS = UserPasswordCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="reddit",
    username=SecretStr("mock-reddit-username"),
    password=SecretStr("mock-reddit-password"),
    title="Mock Reddit credentials",
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


def get_praw(creds: RedditCredentials) -> praw.Reddit:
    client = praw.Reddit(
        client_id=settings.secrets.reddit_client_id,
        client_secret=settings.secrets.reddit_client_secret,
        username=creds.username.get_secret_value(),
        password=creds.password.get_secret_value(),
        user_agent=settings.config.reddit_user_agent,
    )
    me = client.user.me()
    if not me:
        raise ValueError("Invalid Reddit credentials.")
    print(f"Logged in as Reddit user: {me.name}")
    return client


class GetRedditPostsBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
        post: RedditPost = SchemaField(description="Reddit post")

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

    def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        current_time = datetime.now(tz=timezone.utc)
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

            yield "post", RedditPost(
                id=post.id,
                subreddit=input_data.subreddit,
                title=post.title,
                body=post.selftext,
            )


class PostRedditCommentBlock(Block):
    class Input(BlockSchema):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        data: RedditComment = SchemaField(description="Reddit comment")

    class Output(BlockSchema):
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
    def reply_post(creds: RedditCredentials, comment: RedditComment) -> str:
        client = get_praw(creds)
        submission = client.submission(id=comment.post_id)
        new_comment = submission.reply(comment.comment)
        if not new_comment:
            raise ValueError("Failed to post comment.")
        return new_comment.id

    def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        yield "comment_id", self.reply_post(credentials, input_data.data)
