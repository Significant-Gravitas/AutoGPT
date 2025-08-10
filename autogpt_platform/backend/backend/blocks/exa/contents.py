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

from ._config import exa
from .helpers import ContentSettings


class ExaContentsBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        ids: list[str] = SchemaField(
            description="Array of document IDs obtained from searches"
        )
        contents: ContentSettings = SchemaField(
            description="Content retrieval settings",
            default=ContentSettings(),
            advanced=True,
        )

    class Output(BlockSchema):
        results: list = SchemaField(
            description="List of document contents", default_factory=list
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="c52be83f-f8cd-4180-b243-af35f986b461",
            description="Retrieves document contents using Exa's contents API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaContentsBlock.Input,
            output_schema=ExaContentsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/contents"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        # Convert ContentSettings to API format
        payload = {
            "ids": input_data.ids,
            "text": input_data.contents.text,
            "highlights": input_data.contents.highlights,
            "summary": input_data.contents.summary,
        }

        try:
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()
            yield "results", data.get("results", [])
        except Exception as e:
            yield "error", str(e)
