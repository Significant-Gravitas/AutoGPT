# type: ignore

from datetime import datetime, timedelta
from typing import ClassVar

import praw
from pydantic import BaseModel

from autogpt_server.data.block import Block, BlockOutput, BlockSchema


class RedditCredentials(BaseModel):
    client_id: str
    client_secret: str
    username: str
    password: str
    user_agent: str


class RedditPost(BaseModel):
    id: str
    subreddit: str
    title: str
    body: str


def get_praw(creds: RedditCredentials) -> praw.Reddit:
    return praw.Reddit(
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        user_agent=creds.user_agent,
        username=creds.username,
        password=creds.password,
    )


class RedditCredentialsBlock(Block):
    class Input(BlockSchema, RedditCredentials):
        pass

    class Output(BlockSchema):
        reddit: RedditCredentials

    def __init__(self):
        super().__init__(
            id="0c391c35-be7f-40e1-ac27-c3009391f244",
            input_schema=RedditCredentialsBlock.Input,
            output_schema=RedditCredentialsBlock.Output,
        )

    def run(self, input_data: RedditCredentials) -> BlockOutput:
        client = get_praw(input_data)
        client.user.me()
        yield "credential", input_data


class RedditGetPostsBlock(Block):
    class Input(BlockSchema):
        creds: RedditCredentials
        subreddit: str
        last_minutes: int
        last_post: str | None

    class Output(BlockSchema):
        post: RedditPost

    def __init__(self):
        super().__init__(
            id="c6731acb-4285-4ee1-bc9b-03d0766c370f",
            input_schema=RedditGetPostsBlock.Input,
            output_schema=RedditGetPostsBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        client = get_praw(input_data.creds)
        subreddit = client.subreddit(input_data.subreddit)
        oldest_time = datetime.now() - timedelta(minutes=input_data.last_minutes)

        for post in subreddit.new(limit=None):
            if post.created_utc < oldest_time.timestamp():
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
        creds: RedditCredentials
        post_id: str
        comment: str

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
        submission = client.submission(id=input_data.post_id)
        comment = submission.reply(input_data.comment)
        yield "comment_id", comment.id
