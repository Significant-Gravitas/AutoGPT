"""
Meeting BaaS bot (recording) blocks.
"""

from typing import Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import MeetingBaasAPI
from ._config import baas


class BaasBotJoinMeetingBlock(Block):
    """
    Deploy a bot immediately or at a scheduled start_time to join and record a meeting.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        meeting_url: str = SchemaField(
            description="The URL of the meeting the bot should join"
        )
        bot_name: str = SchemaField(
            description="Display name for the bot in the meeting"
        )
        bot_image: str = SchemaField(
            description="URL to an image for the bot's avatar (16:9 ratio recommended)",
            default="",
        )
        entry_message: str = SchemaField(
            description="Chat message the bot will post upon entry", default=""
        )
        reserved: bool = SchemaField(
            description="Use a reserved bot slot (joins 4 min before meeting)",
            default=False,
        )
        start_time: Optional[int] = SchemaField(
            description="Unix timestamp (ms) when bot should join", default=None
        )
        webhook_url: str | None = SchemaField(
            description="URL to receive webhook events for this bot", default=None
        )
        timeouts: dict = SchemaField(
            description="Automatic leave timeouts configuration", default={}
        )
        extra: dict = SchemaField(
            description="Custom metadata to attach to the bot", default={}
        )

    class Output(BlockSchemaOutput):
        bot_id: str = SchemaField(description="UUID of the deployed bot")
        join_response: dict = SchemaField(
            description="Full response from join operation"
        )

    def __init__(self):
        super().__init__(
            id="377d1a6a-a99b-46cf-9af3-1d1b12758e04",
            description="Deploy a bot to join and record a meeting",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Call API with all parameters
        data = await api.join_meeting(
            bot_name=input_data.bot_name,
            meeting_url=input_data.meeting_url,
            reserved=input_data.reserved,
            bot_image=input_data.bot_image if input_data.bot_image else None,
            entry_message=(
                input_data.entry_message if input_data.entry_message else None
            ),
            start_time=input_data.start_time,
            speech_to_text={"provider": "Default"},
            webhook_url=input_data.webhook_url if input_data.webhook_url else None,
            automatic_leave=input_data.timeouts if input_data.timeouts else None,
            extra=input_data.extra if input_data.extra else None,
        )

        yield "bot_id", data.get("bot_id", "")
        yield "join_response", data


class BaasBotLeaveMeetingBlock(Block):
    """
    Force the bot to exit the call.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        bot_id: str = SchemaField(description="UUID of the bot to remove from meeting")

    class Output(BlockSchemaOutput):
        left: bool = SchemaField(description="Whether the bot successfully left")

    def __init__(self):
        super().__init__(
            id="bf77d128-8b25-4280-b5c7-2d553ba7e482",
            description="Remove a bot from an ongoing meeting",
            categories={BlockCategory.COMMUNICATION},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Leave meeting
        left = await api.leave_meeting(input_data.bot_id)

        yield "left", left


class BaasBotFetchMeetingDataBlock(Block):
    """
    Pull MP4 URL, transcript & metadata for a completed meeting.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        bot_id: str = SchemaField(description="UUID of the bot whose data to fetch")
        include_transcripts: bool = SchemaField(
            description="Include transcript data in response", default=True
        )

    class Output(BlockSchemaOutput):
        mp4_url: str = SchemaField(
            description="URL to download the meeting recording (time-limited)"
        )
        transcript: list = SchemaField(description="Meeting transcript data")
        metadata: dict = SchemaField(description="Meeting metadata and bot information")

    def __init__(self):
        super().__init__(
            id="ea7c1309-303c-4da1-893f-89c0e9d64e78",
            description="Retrieve recorded meeting data",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Fetch meeting data
        data = await api.get_meeting_data(
            bot_id=input_data.bot_id,
            include_transcripts=input_data.include_transcripts,
        )

        yield "mp4_url", data.get("mp4", "")
        yield "transcript", data.get("bot_data", {}).get("transcripts", [])
        yield "metadata", data.get("bot_data", {}).get("bot", {})


class BaasBotDeleteRecordingBlock(Block):
    """
    Purge MP4 + transcript data for privacy or storage management.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        bot_id: str = SchemaField(description="UUID of the bot whose data to delete")

    class Output(BlockSchemaOutput):
        deleted: bool = SchemaField(
            description="Whether the data was successfully deleted"
        )

    def __init__(self):
        super().__init__(
            id="bf8d1aa6-42d8-4944-b6bd-6bac554c0d3b",
            description="Permanently delete a meeting's recorded data",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Delete recording data
        deleted = await api.delete_data(input_data.bot_id)

        yield "deleted", deleted
