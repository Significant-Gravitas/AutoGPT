# type: ignore

from datetime import datetime, timedelta, timezone

import praw
from typing import Any
from pydantic import BaseModel, Field

from autogpt_server.data.block import Block, BlockOutput, BlockSchema


class RedditCredentials(BaseModel):
    client_id: str
    client_secret: str
    username: str
    password: str
    user_agent: str | None = None


class RedditPost(BaseModel):
    id: str
    subreddit: str
    title: str
    body: str


def get_praw(creds: RedditCredentials) -> praw.Reddit:
    client = praw.Reddit(
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        user_agent=creds.user_agent,
        username=creds.username,
        password=creds.password,
    )
    me = client.user.me()
    if not me:
        raise ValueError("Invalid Reddit credentials.")
    print(f"Logged in as Reddit user: {me.name}")
    return client


class RedditGetPostsBlock(Block):
    class Input(BlockSchema):
        creds: RedditCredentials = Field(description="Reddit credentials")
        subreddit: str = Field(description="Subreddit name")
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
        )

    def run(self, input_data: Input) -> BlockOutput:
        client = get_praw(input_data.creds)
        subreddit = client.subreddit(input_data.subreddit)
        for post in subreddit.new(limit=input_data.post_limit):
            if input_data.last_post and post.created_utc < datetime.now(tz=timezone.utc) - \
                    timedelta(minutes=input_data.last_minutes):
                break

            if input_data.last_post and post.id == input_data.last_post:
                break

            yield "post", RedditPost(
                id=post.id,
                subreddit=subreddit.display_name,
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
