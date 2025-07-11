"""
ElevenLabs voice management blocks.
"""

from typing import Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import elevenlabs


class ElevenLabsListVoicesBlock(Block):
    """
    Fetch all voices the account can use (for pick-lists, UI menus, etc.).
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        search: str = SchemaField(
            description="Search term to filter voices", default=""
        )
        voice_type: Optional[str] = SchemaField(
            description="Filter by voice type: premade, cloned, or professional",
            default=None,
        )
        page_size: int = SchemaField(
            description="Number of voices per page (max 100)", default=10
        )
        next_page_token: str = SchemaField(
            description="Token for fetching next page", default=""
        )

    class Output(BlockSchema):
        voices: list[dict] = SchemaField(
            description="Array of voice objects with id, name, category, etc."
        )
        next_page_token: Optional[str] = SchemaField(
            description="Token for fetching next page, null if no more pages"
        )

    def __init__(self):
        super().__init__(
            id="e1a2b3c4-d5e6-f7a8-b9c0-d1e2f3a4b5c6",
            description="List all available voices with filtering and pagination",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build query parameters
        params: dict[str, str | int] = {"page_size": input_data.page_size}

        if input_data.search:
            params["search"] = input_data.search
        if input_data.voice_type:
            params["voice_type"] = input_data.voice_type
        if input_data.next_page_token:
            params["next_page_token"] = input_data.next_page_token

        # Fetch voices
        response = await Requests().get(
            "https://api.elevenlabs.io/v2/voices",
            headers={"xi-api-key": api_key},
            params=params,
        )

        data = response.json()

        yield "voices", data.get("voices", [])
        yield "next_page_token", data.get("next_page_token")


class ElevenLabsGetVoiceDetailsBlock(Block):
    """
    Retrieve metadata/settings for a single voice.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        voice_id: str = SchemaField(description="The ID of the voice to retrieve")

    class Output(BlockSchema):
        voice: dict = SchemaField(
            description="Voice object with name, labels, settings, etc."
        )

    def __init__(self):
        super().__init__(
            id="f2a3b4c5-d6e7-f8a9-b0c1-d2e3f4a5b6c7",
            description="Get detailed information about a specific voice",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Fetch voice details
        response = await Requests().get(
            f"https://api.elevenlabs.io/v1/voices/{input_data.voice_id}",
            headers={"xi-api-key": api_key},
        )

        voice = response.json()

        yield "voice", voice


class ElevenLabsCreateVoiceCloneBlock(Block):
    """
    Upload sample clips to create a custom (IVC) voice.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        name: str = SchemaField(description="Name for the new voice")
        files: list[str] = SchemaField(
            description="Base64-encoded audio files (1-10 files, max 25MB each)"
        )
        description: str = SchemaField(
            description="Description of the voice", default=""
        )
        labels: dict = SchemaField(
            description="Metadata labels (e.g., accent, age)", default={}
        )
        remove_background_noise: bool = SchemaField(
            description="Whether to remove background noise from samples", default=False
        )

    class Output(BlockSchema):
        voice_id: str = SchemaField(description="ID of the newly created voice")
        requires_verification: bool = SchemaField(
            description="Whether the voice requires verification"
        )

    def __init__(self):
        super().__init__(
            id="a3b4c5d6-e7f8-a9b0-c1d2-e3f4a5b6c7d8",
            description="Create a new voice clone from audio samples",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        import base64
        import json
        from io import BytesIO

        api_key = credentials.api_key.get_secret_value()

        # Prepare multipart form data
        form_data = {
            "name": input_data.name,
        }

        if input_data.description:
            form_data["description"] = input_data.description
        if input_data.labels:
            form_data["labels"] = json.dumps(input_data.labels)
        if input_data.remove_background_noise:
            form_data["remove_background_noise"] = "true"

        # Prepare files
        files = []
        for i, file_b64 in enumerate(input_data.files):
            file_data = base64.b64decode(file_b64)
            files.append(
                ("files", (f"sample_{i}.mp3", BytesIO(file_data), "audio/mpeg"))
            )

        # Create voice
        response = await Requests().post(
            "https://api.elevenlabs.io/v1/voices/add",
            headers={"xi-api-key": api_key},
            data=form_data,
            files=files,
        )

        result = response.json()

        yield "voice_id", result.get("voice_id", "")
        yield "requires_verification", result.get("requires_verification", False)


class ElevenLabsDeleteVoiceBlock(Block):
    """
    Permanently remove a custom voice.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        voice_id: str = SchemaField(description="The ID of the voice to delete")

    class Output(BlockSchema):
        status: str = SchemaField(description="Deletion status (ok or error)")

    def __init__(self):
        super().__init__(
            id="b4c5d6e7-f8a9-b0c1-d2e3-f4a5b6c7d8e9",
            description="Delete a custom voice from your account",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Delete voice
        response = await Requests().delete(
            f"https://api.elevenlabs.io/v1/voices/{input_data.voice_id}",
            headers={"xi-api-key": api_key},
        )

        # Check if successful
        if response.status in [200, 204]:
            yield "status", "ok"
        else:
            yield "status", "error"
