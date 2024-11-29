from datetime import datetime
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


class ContentSettings(BaseModel):
    text: dict = SchemaField(
        description="Text content settings",
        default={"maxCharacters": 1000, "includeHtmlTags": False},
    )
    highlights: dict = SchemaField(
        description="Highlight settings",
        default={"numSentences": 3, "highlightsPerUrl": 3},
    )
    summary: dict = SchemaField(
        description="Summary settings",
        default={"query": ""},
    )


class ExaSearchBlock(Block):
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        query: str = SchemaField(description="The search query")
        useAutoprompt: bool = SchemaField(
            description="Whether to use autoprompt",
            default=True,
        )
        type: str = SchemaField(
            description="Type of search",
            default="",
        )
        category: str = SchemaField(
            description="Category to search within",
            default="",
        )
        numResults: int = SchemaField(
            description="Number of results to return",
            default=10,
        )
        includeDomains: List[str] = SchemaField(
            description="Domains to include in search",
            default=[],
        )
        excludeDomains: List[str] = SchemaField(
            description="Domains to exclude from search",
            default=[],
        )
        startCrawlDate: datetime = SchemaField(
            description="Start date for crawled content",
        )
        endCrawlDate: datetime = SchemaField(
            description="End date for crawled content",
        )
        startPublishedDate: datetime = SchemaField(
            description="Start date for published content",
        )
        endPublishedDate: datetime = SchemaField(
            description="End date for published content",
        )
        includeText: List[str] = SchemaField(
            description="Text patterns to include",
            default=[],
        )
        excludeText: List[str] = SchemaField(
            description="Text patterns to exclude",
            default=[],
        )
        contents: ContentSettings = SchemaField(
            description="Content retrieval settings",
            default=ContentSettings(),
        )

    class Output(BlockSchema):
        results: list = SchemaField(
            description="List of search results",
            default=[],
        )

    def __init__(self):
        super().__init__(
            id="996cec64-ac40-4dde-982f-b0dc60a5824d",
            description="Searches the web using Exa's advanced search API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaSearchBlock.Input,
            output_schema=ExaSearchBlock.Output,
        )

    def run(
        self, input_data: Input, *, credentials: ExaCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/search"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        payload = {
            "query": input_data.query,
            "useAutoprompt": input_data.useAutoprompt,
            "numResults": input_data.numResults,
            "contents": {
                "text": {"maxCharacters": 1000, "includeHtmlTags": False},
                "highlights": {
                    "numSentences": 3,
                    "highlightsPerUrl": 3,
                },
                "summary": {"query": ""},
            },
        }

        # Add dates if they exist
        date_fields = [
            "startCrawlDate",
            "endCrawlDate",
            "startPublishedDate",
            "endPublishedDate",
        ]
        for field in date_fields:
            value = getattr(input_data, field, None)
            if value:
                payload[field] = value.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Add other fields
        optional_fields = [
            "type",
            "category",
            "includeDomains",
            "excludeDomains",
            "includeText",
            "excludeText",
        ]

        for field in optional_fields:
            value = getattr(input_data, field)
            if value:  # Only add non-empty values
                payload[field] = value

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            # Extract just the results array from the response
            yield "results", data.get("results", [])
        except Exception as e:
            yield "error", str(e)
            yield "results", []
