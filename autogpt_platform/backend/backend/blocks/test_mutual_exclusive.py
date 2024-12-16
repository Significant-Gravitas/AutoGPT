from typing import List, Union

from pydantic import BaseModel
from typing_extensions import Literal

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class PollOption(BaseModel):
    text: str


class Poll(BaseModel):
    discriminator: Literal["poll"]
    some_input_4: List[PollOption]


class MediaUpload(BaseModel):
    discriminator: Literal["media"]
    some_input: str
    some_input_2: str


class PollDuration(BaseModel):
    discriminator: Literal["duration"]
    some_input: int


class TweetBlock(Block):
    class Input(BlockSchema):
        tweet_text: str = SchemaField(
            title="Tweet Text",
            description="The main text content of the tweet",
        )

        attachment: Union[Poll, MediaUpload, PollDuration] = SchemaField(
            discriminator="discriminator",
            title="Tweet Attachment",
            description="Optional tweet attachment (poll, media, or duration)",
        )

    class Output(BlockSchema):
        result: str = SchemaField(
            description="Shows the tweet content and any attachments"
        )

    def __init__(self):
        super().__init__(
            id="b7faa910-b074-11ef-bee7-477f51db4711",
            description="Create a tweet with optional attachments",
            categories={BlockCategory.BASIC},
            input_schema=TweetBlock.Input,
            output_schema=TweetBlock.Output,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        tweet_content = [f"Tweet Text: {input_data.tweet_text}"]
        if isinstance(input_data.attachment, Poll):
            options = [opt.text for opt in input_data.attachment.some_input_4]
            tweet_content.append(f"Poll Options: {', '.join(options)}")

        if isinstance(input_data.attachment, MediaUpload):
            tweet_content.append(f"Media URL: {input_data.attachment.some_input}")
            tweet_content.append(f"Media URL 2: {input_data.attachment.some_input_2}")

        if isinstance(input_data.attachment, PollDuration):
            tweet_content.append(
                f"Poll Duration: {input_data.attachment.some_input} hours"
            )

        yield "result", "\n".join(tweet_content)
