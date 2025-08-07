"""
Meeting BaaS bot (recording) blocks.
"""

from typing import Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import MeetingBaasAPI
from ._config import baas


class BaasBotJoinMeetingBlock(Block):
    """
    Deploy a bot immediately or at a scheduled start_time to join and record a meeting.
    """

    class Input(BlockSchema):
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
        speech_to_text: dict = SchemaField(
            description="Speech-to-text configuration", default={"provider": "Gladia"}
        )
        webhook_url: str = SchemaField(
            description="URL to receive webhook events for this bot", default=""
        )
        timeouts: dict = SchemaField(
            description="Automatic leave timeouts configuration", default={}
        )
        extra: dict = SchemaField(
            description="Custom metadata to attach to the bot", default={}
        )

    class Output(BlockSchema):
        bot_id: str = SchemaField(description="UUID of the deployed bot")
        join_response: dict = SchemaField(
            description="Full response from join operation"
        )

    def __init__(self):
        super().__init__(
            id="7f8e9d0c-1b2a-3c4d-5e6f-7a8b9c0d1e2f",
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
            speech_to_text=input_data.speech_to_text,
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

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        bot_id: str = SchemaField(description="UUID of the bot to remove from meeting")

    class Output(BlockSchema):
        left: bool = SchemaField(description="Whether the bot successfully left")

    def __init__(self):
        super().__init__(
            id="8a9b0c1d-2e3f-4a5b-6c7d-8e9f0a1b2c3d",
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

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        bot_id: str = SchemaField(description="UUID of the bot whose data to fetch")
        include_transcripts: bool = SchemaField(
            description="Include transcript data in response", default=True
        )

    class Output(BlockSchema):
        mp4_url: str = SchemaField(
            description="URL to download the meeting recording (time-limited)"
        )
        transcript: list = SchemaField(description="Meeting transcript data")
        metadata: dict = SchemaField(description="Meeting metadata and bot information")

    def __init__(self):
        super().__init__(
            id="9b0c1d2e-3f4a-5b6c-7d8e-9f0a1b2c3d4e",
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


class BaasBotFetchScreenshotsBlock(Block):
    """
    List screenshots captured during the call.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        bot_id: str = SchemaField(
            description="UUID of the bot whose screenshots to fetch"
        )

    class Output(BlockSchema):
        screenshots: list[dict] = SchemaField(
            description="Array of screenshot objects with date and url"
        )

    def __init__(self):
        super().__init__(
            id="0c1d2e3f-4a5b-6c7d-8e9f-0a1b2c3d4e5f",
            description="Retrieve screenshots captured during a meeting",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Fetch screenshots
        screenshots = await api.get_screenshots(input_data.bot_id)

        yield "screenshots", screenshots


class BaasBotDeleteRecordingBlock(Block):
    """
    Purge MP4 + transcript data for privacy or storage management.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        bot_id: str = SchemaField(description="UUID of the bot whose data to delete")

    class Output(BlockSchema):
        deleted: bool = SchemaField(
            description="Whether the data was successfully deleted"
        )

    def __init__(self):
        super().__init__(
            id="1d2e3f4a-5b6c-7d8e-9f0a-1b2c3d4e5f6a",
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


class BaasBotRetranscribeBlock(Block):
    """
    Re-run STT on past audio with a different provider or settings.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        bot_id: str = SchemaField(
            description="UUID of the bot whose audio to retranscribe"
        )
        provider: str = SchemaField(
            description="Speech-to-text provider to use (e.g., Gladia, Deepgram)"
        )
        webhook_url: str = SchemaField(
            description="URL to receive transcription complete event", default=""
        )
        custom_options: dict = SchemaField(
            description="Provider-specific options", default={}
        )

    class Output(BlockSchema):
        job_id: Optional[str] = SchemaField(
            description="Transcription job ID if available"
        )
        accepted: bool = SchemaField(
            description="Whether the retranscription request was accepted"
        )

    def __init__(self):
        super().__init__(
            id="2e3f4a5b-6c7d-8e9f-0a1b-2c3d4e5f6a7b",
            description="Re-run transcription on a meeting's audio",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        api = MeetingBaasAPI(api_key)

        # Build speech_to_text config from provider and custom options
        speech_to_text = {"provider": input_data.provider}
        if input_data.custom_options:
            speech_to_text.update(input_data.custom_options)

        # Start retranscription
        result = await api.retranscribe(
            bot_uuid=input_data.bot_id,
            speech_to_text=speech_to_text,
            webhook_url=input_data.webhook_url if input_data.webhook_url else None,
        )

        # Check if accepted
        accepted = result.get("accepted", False) or "job_id" in result
        job_id = result.get("job_id")

        yield "job_id", job_id
        yield "accepted", accepted
