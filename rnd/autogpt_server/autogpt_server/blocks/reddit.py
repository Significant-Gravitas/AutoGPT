# type: ignore

from datetime import datetime, timedelta, timezone

import praw
from typing import Any
from pydantic import BaseModel, Field
from typing import Iterator

from autogpt_server.data.block import Block, BlockOutput, BlockSchema, BlockFieldSecret
from autogpt_server.util.mock import MockObject


class RedditCredentials(BaseModel):
    client_id: BlockFieldSecret = BlockFieldSecret(key="reddit_client_id")
    client_secret: BlockFieldSecret = BlockFieldSecret(key="reddit_client_secret")
    username: BlockFieldSecret = BlockFieldSecret(key="reddit_username")
    password: BlockFieldSecret = BlockFieldSecret(key="reddit_password")
    user_agent: str | None = None


class RedditPost(BaseModel):
    id: str
    subreddit: str
    title: str
    body: str


def get_praw(creds: RedditCredentials) -> praw.Reddit:
    client = praw.Reddit(
        client_id=creds.client_id.get(),
        client_secret=creds.client_secret.get(),
        username=creds.username.get(),
        password=creds.password.get(),
        user_agent=creds.user_agent,
    )
    me = client.user.me()
    if not me:
        raise ValueError("Invalid Reddit credentials.")
    print(f"Logged in as Reddit user: {me.name}")
    return client


class RedditGetPostsBlock(Block):
    class Input(BlockSchema):
        subreddit: str = Field(description="Subreddit name")
        creds: RedditCredentials = Field(
            description="Reddit credentials",
            default=RedditCredentials(),
        )
        last_minutes: int | None = Field(
            description="Post time to stop minutes ago while fetching posts",
            default=None
        )
        last_post: str | None = Field(
            description="Post ID to stop when reached while fetching posts",
            default=None
        )
        post_limit: int | None = Field(
            description="Number of posts to fetch",
            default=10
        )

    class Output(BlockSchema):
        post: RedditPost = Field(description="Reddit post")

    def __init__(self):
        super().__init__(
            id="c6731acb-4285-4ee1-bc9b-03d0766c370f",
            input_schema=RedditGetPostsBlock.Input,
            output_schema=RedditGetPostsBlock.Output,
            test_input={
                "creds": {
                    "client_id": "client_id",
                    "client_secret": "client_secret",
                    "username": "username",
                    "password": "password",
                    "user_agent": "user_agent",
                },
                "subreddit": "subreddit",
                "last_post": "id3",
                "post_limit": 2,
            },
            test_output=[
                ("post", RedditPost(
                    id="id1", subreddit="subreddit", title="title1", body="body1")),
                ("post", RedditPost(
                    id="id2", subreddit="subreddit", title="title2", body="body2")),
            ],
            test_mock={
                "get_posts": lambda _: [
                    MockObject(id="id1", title="title1", selftext="body1"),
                    MockObject(id="id2", title="title2", selftext="body2"),
                    MockObject(id="id3", title="title2", selftext="body2"),
                ]
            }
        )

    @staticmethod
    def get_posts(input_data: Input) -> Iterator[praw.reddit.Submission]:
        client = get_praw(input_data.creds)
        subreddit = client.subreddit(input_data.subreddit)
        return subreddit.new(limit=input_data.post_limit)

    def run(self, input_data: Input) -> BlockOutput:
        for post in self.get_posts(input_data):
            if input_data.last_minutes and post.created_utc < datetime.now(
                    tz=timezone.utc) - \
                    timedelta(minutes=input_data.last_minutes):
                break

            if input_data.last_post and post.id == input_data.last_post:
                break

            yield "post", RedditPost(
                id=post.id,
                subreddit=input_data.subreddit,
                title=post.title,
                body=post.selftext
            )


class RedditPostCommentBlock(Block):
    class Input(BlockSchema):
        creds: RedditCredentials = Field(description="Reddit credentials")
        data: Any = Field(description="Reddit post")
        # post_id: str = Field(description="Reddit post ID")
        # comment: str = Field(description="Comment text")

    class Output(BlockSchema):
        comment_id: str

    def __init__(self):
        super().__init__(
            id="4a92261b-701e-4ffb-8970-675fd28e261f",
            input_schema=RedditPostCommentBlock.Input,
            output_schema=RedditPostCommentBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        client = get_praw(input_data.creds)
        submission = client.submission(id=input_data.data["post_id"])
        comment = submission.reply(input_data.data["comment"])
        yield "comment_id", comment.id
