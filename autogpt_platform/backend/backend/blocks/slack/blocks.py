"""
Slack action blocks for sending messages to channels and DMs.
"""

import logging
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import APIKeyCredentials, SchemaField

from ._api import SlackMessageResult, post_message
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    SlackCredentialsField,
    SlackCredentialsInput,
)

logger = logging.getLogger(__name__)


class SendSlackMessageBlock(Block):
    """Send a text message to a Slack channel or DM."""

    class Input(BlockSchemaInput):
        credentials: SlackCredentialsInput = SlackCredentialsField()
        channel: str = SchemaField(
            description="Channel ID (e.g. C1234567890) or name (e.g. #general). "
            "For DMs use the user's member ID (e.g. U1234567890)."
        )
        text: str = SchemaField(
            description="Message text. Supports Slack mrkdwn: "
            "*bold*, _italic_, `code`, <https://example.com|link>."
        )
        thread_ts: Optional[str] = SchemaField(
            description="Timestamp of the parent message to reply in a thread. "
            "Use the 'ts' output from a previous Send Slack Message block.",
            default=None,
            advanced=True,
        )
        username: Optional[str] = SchemaField(
            description="Custom display name for the bot in this message. "
            "Requires chat:write.customize scope.",
            default=None,
            advanced=True,
        )
        icon_emoji: Optional[str] = SchemaField(
            description="Emoji to use as the bot's icon (e.g. :robot_face:). "
            "Requires chat:write.customize scope.",
            default=None,
            advanced=True,
        )
        unfurl_links: bool = SchemaField(
            description="Automatically expand URLs into rich previews.",
            default=True,
            advanced=True,
        )
        mrkdwn: bool = SchemaField(
            description="Enable Slack markdown formatting in the message text.",
            default=True,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        ts: str = SchemaField(
            description="Message timestamp — Slack's unique message ID. "
            "Use as 'thread_ts' to reply in a thread."
        )
        channel: str = SchemaField(
            description="The channel ID where the message was posted."
        )

    def __init__(self):
        super().__init__(
            id="50cf5290-ec87-4ca1-b7d7-b8b4ef772f6d",
            description="Send a text message to any Slack channel, DM, or thread. "
            "Required bot token scope: chat:write.",
            categories={BlockCategory.SOCIAL},
            input_schema=SendSlackMessageBlock.Input,
            output_schema=SendSlackMessageBlock.Output,
            test_input={
                "channel": "C1234567890",
                "text": "Hello from AutoGPT!",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ts", "1234567890.123456"),
                ("channel", "C1234567890"),
            ],
            test_mock={
                "_post_message": lambda *args, **kwargs: SlackMessageResult(
                    ts="1234567890.123456",
                    channel="C1234567890",
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        result = await self._post_message(
            credentials=credentials,
            channel=input_data.channel,
            text=input_data.text,
            thread_ts=input_data.thread_ts,
            username=input_data.username,
            icon_emoji=input_data.icon_emoji,
            unfurl_links=input_data.unfurl_links,
            mrkdwn=input_data.mrkdwn,
        )
        yield "ts", result.ts
        yield "channel", result.channel

    async def _post_message(
        self,
        credentials: APIKeyCredentials,
        channel: str,
        text: str,
        thread_ts: Optional[str],
        username: Optional[str],
        icon_emoji: Optional[str],
        unfurl_links: bool,
        mrkdwn: bool,
    ) -> SlackMessageResult:
        return await post_message(
            credentials=credentials,
            channel=channel,
            text=text,
            thread_ts=thread_ts,
            username=username,
            icon_emoji=icon_emoji,
            unfurl_links=unfurl_links,
            mrkdwn=mrkdwn,
        )
