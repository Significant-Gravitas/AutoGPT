from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    List,
    SchemaField,
    String,
    Requests,
)

from ._config import exa
from .helpers import ContentSettings


class ExaContentsBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        ids: List[String] = SchemaField(
            description="Array of document IDs obtained from searches"
        )
        contents: ContentSettings = SchemaField(
            description="Content retrieval settings",
            default=ContentSettings(),
            advanced=True,
        )

    class Output(BlockSchema):
        results: List = SchemaField(
            description="List of document contents", default_factory=list
        )
        error: String = SchemaField(
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
        contents_dict = input_data.contents.model_dump()
        payload = {
            "ids": input_data.ids,
            "text": {
                "maxCharacters": contents_dict["text"]["max_characters"],
                "includeHtmlTags": contents_dict["text"]["include_html_tags"],
            },
            "highlights": {
                "numSentences": contents_dict["highlights"]["num_sentences"],
                "highlightsPerUrl": contents_dict["highlights"]["highlights_per_url"],
                "query": contents_dict["summary"][
                    "query"
                ],  # Note: query comes from summary
            },
            "summary": {
                "query": contents_dict["summary"]["query"],
            },
        }

        try:
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()
            yield "results", data.get("results", [])
        except Exception as e:
            yield "error", str(e)
