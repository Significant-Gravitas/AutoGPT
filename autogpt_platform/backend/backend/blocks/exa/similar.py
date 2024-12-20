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

from .helpers import ContentSettings


class ExaFindSimilarBlock(Block):
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        url: str = SchemaField(
            description="The url for which you would like to find similar links"
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
            description="Text patterns to include (max 1 string, up to 5 words)",
            default=[],
        )
        excludeText: List[str] = SchemaField(
            description="Text patterns to exclude (max 1 string, up to 5 words)",
            default=[],
        )
        contents: ContentSettings = SchemaField(
            description="Content retrieval settings",
            default=ContentSettings(),
        )

    class Output(BlockSchema):
        results: list = SchemaField(
            description="List of similar documents with title, URL, published date, author, and score",
            default=[],
        )

    def __init__(self):
        super().__init__(
            id="5e7315d1-af61-4a0c-9350-7c868fa7438a",
            description="Finds similar links using Exa's findSimilar API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaFindSimilarBlock.Input,
            output_schema=ExaFindSimilarBlock.Output,
        )

    def run(
        self, input_data: Input, *, credentials: ExaCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/findSimilar"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        payload = {
            "url": input_data.url,
            "numResults": input_data.numResults,
            "contents": {
                "text": input_data.contents.text,
                "highlights": input_data.contents.highlights,
                "summary": input_data.contents.summary,
            },
        }

        # Add optional fields if they have values
        optional_fields = [
            "includeDomains",
            "excludeDomains",
            "includeText",
            "excludeText",
        ]
        for field in optional_fields:
            value = getattr(input_data, field)
            if value:  # Only add non-empty values
                payload[field] = value

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

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            yield "results", data.get("results", [])
        except Exception as e:
            yield "error", str(e)
            yield "results", []