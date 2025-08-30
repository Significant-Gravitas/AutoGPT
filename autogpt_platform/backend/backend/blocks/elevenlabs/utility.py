"""
ElevenLabs utility blocks for models and usage stats.
"""

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


class ElevenLabsListModelsBlock(Block):
    """
    Get all available model IDs & capabilities.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )

    class Output(BlockSchema):
        models: list[dict] = SchemaField(
            description="Array of model objects with capabilities"
        )

    def __init__(self):
        super().__init__(
            id="a9b0c1d2-e3f4-a5b6-c7d8-e9f0a1b2c3d4",
            description="List all available voice models and their capabilities",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Fetch models
        response = await Requests().get(
            "https://api.elevenlabs.io/v1/models",
            headers={"xi-api-key": api_key},
        )

        models = response.json()

        yield "models", models


class ElevenLabsGetUsageStatsBlock(Block):
    """
    Character / credit usage for billing dashboards.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = elevenlabs.credentials_field(
            description="ElevenLabs API credentials"
        )
        start_unix: int = SchemaField(
            description="Start timestamp in Unix epoch seconds"
        )
        end_unix: int = SchemaField(description="End timestamp in Unix epoch seconds")
        aggregation_interval: str = SchemaField(
            description="Aggregation interval: daily or monthly",
            default="daily",
        )

    class Output(BlockSchema):
        usage: list[dict] = SchemaField(description="Array of usage data per interval")
        total_character_count: int = SchemaField(
            description="Total characters used in period"
        )
        total_requests: int = SchemaField(description="Total API requests in period")

    def __init__(self):
        super().__init__(
            id="b0c1d2e3-f4a5-b6c7-d8e9-f0a1b2c3d4e5",
            description="Get character and credit usage statistics",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Build query parameters
        params = {
            "start_unix": input_data.start_unix,
            "end_unix": input_data.end_unix,
            "aggregation_interval": input_data.aggregation_interval,
        }

        # Fetch usage stats
        response = await Requests().get(
            "https://api.elevenlabs.io/v1/usage/character-stats",
            headers={"xi-api-key": api_key},
            params=params,
        )

        data = response.json()

        yield "usage", data.get("usage", [])
        yield "total_character_count", data.get("total_character_count", 0)
        yield "total_requests", data.get("total_requests", 0)
