from datetime import datetime, timezone
from typing import Iterator

import praw
from pydantic import BaseModel, ConfigDict

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField
from backend.util.mock import MockObject


class RedditCredentials(BaseModel):
    client_id: BlockSecret = SecretField(key="reddit_client_id")
    client_secret: BlockSecret = SecretField(key="reddit_client_secret")
    username: BlockSecret = SecretField(key="reddit_username")
    password: BlockSecret = SecretField(key="reddit_password")
    user_agent: str = "AutoGPT:1.0 (by /u/autogpt)"

    model_config = ConfigDict(title="Reddit Credentials")


class RedditPost(BaseModel):
    id: str
    subreddit: str
    title: str
    body: str


class RedditComment(BaseModel):
    post_id: str
    comment: str


def get_praw(creds: RedditCredentials) -> praw.Reddit:
    client = praw.Reddit(
        client_id=creds.client_id.get_secret_value(),
        client_secret=creds.client_secret.get_secret_value(),
        username=creds.username.get_secret_value(),
        password=creds.password.get_secret_value(),
        user_agent=creds.user_agent,
    )
    me = client.user.me()
    if not me:
        raise ValueError("Invalid Reddit credentials.")
    print(f"Logged in as Reddit user: {me.name}")
    return client


class GetRedditPostsBlock(Block):
    class Input(BlockSchema):
        subreddit: str = SchemaField(description="Subreddit name")
        creds: RedditCredentials = SchemaField(
            description="Reddit credentials",
            default=RedditCredentials(),
        )
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
            disabled=True,
            id="c6731acb-4285-4ee1-bc9b-03d0766c370f",
            description="This block fetches Reddit posts from a defined subreddit name.",
            categories={BlockCategory.SOCIAL},
            input_schema=GetRedditPostsBlock.Input,
            output_schema=GetRedditPostsBlock.Output,
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
                "get_posts": lambda _: [
                    MockObject(id="id1", title="title1", selftext="body1"),
                    MockObject(id="id2", title="title2", selftext="body2"),
                    MockObject(id="id3", title="title2", selftext="body2"),
                ]
            },
        )

    @staticmethod
    def get_posts(input_data: Input) -> Iterator[praw.reddit.Submission]:
        client = get_praw(input_data.creds)
        subreddit = client.subreddit(input_data.subreddit)
        return subreddit.new(limit=input_data.post_limit or 10)

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        current_time = datetime.now(tz=timezone.utc)
        for post in self.get_posts(input_data):
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
        creds: RedditCredentials = SchemaField(
            description="Reddit credentials", default=RedditCredentials()
        )
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
            test_input={"data": {"post_id": "id", "comment": "comment"}},
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

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "comment_id", self.reply_post(input_data.creds, input_data.data)
