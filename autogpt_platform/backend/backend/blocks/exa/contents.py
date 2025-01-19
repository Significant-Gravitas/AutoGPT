from typing import List

from pydantic import BaseModel

from backend.blocks.exa._auth import (
    ExaCredentials,
    ExaCredentialsField,
    ExaCredentialsInput,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.request import requests


class ContentRetrievalSettings(BaseModel):
    text: dict = SchemaField(
        description="Text content settings",
        default={"maxCharacters": 1000, "includeHtmlTags": False},
        advanced=True,
    )
    highlights: dict = SchemaField(
        description="Highlight settings",
        default={
            "numSentences": 3,
            "highlightsPerUrl": 3,
            "query": "",
        },
        advanced=True,
    )
    summary: dict = SchemaField(
        description="Summary settings",
        default={"query": ""},
        advanced=True,
    )


class ExaContentsBlock(Block):
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        ids: List[str] = SchemaField(
            description="Array of document IDs obtained from searches",
        )
        contents: ContentRetrievalSettings = SchemaField(
            description="Content retrieval settings",
            default=ContentRetrievalSettings(),
            advanced=True,
        )

    class Output(BlockSchema):
        results: list = SchemaField(
            description="List of document contents",
            default=[],
        )

    def __init__(self):
        super().__init__(
            id="c52be83f-f8cd-4180-b243-af35f986b461",
            description="Retrieves document contents using Exa's contents API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaContentsBlock.Input,
            output_schema=ExaContentsBlock.Output,
        )

    def run(
        self, input_data: Input, *, credentials: ExaCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/contents"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        payload = {
            "ids": input_data.ids,
            "text": input_data.contents.text,
            "highlights": input_data.contents.highlights,
            "summary": input_data.contents.summary,
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            yield "results", data.get("results", [])
        except Exception as e:
            yield "error", str(e)
            yield "results", []
