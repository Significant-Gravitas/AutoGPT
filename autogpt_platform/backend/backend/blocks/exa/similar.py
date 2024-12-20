from datetime import datetime
from typing import Any, List

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
        number_of_results: int = SchemaField(
            description="Number of results to return",
            default=10,
            advanced=True,
        )
        include_domains: List[str] = SchemaField(
            description="Domains to include in search",
            default=[],
            advanced=True,
        )
        exclude_domains: List[str] = SchemaField(
            description="Domains to exclude from search",
            default=[],
            advanced=True,
        )
        start_crawl_date: datetime = SchemaField(
            description="Start date for crawled content",
        )
        end_crawl_date: datetime = SchemaField(
            description="End date for crawled content",
        )
        start_published_date: datetime = SchemaField(
            description="Start date for published content",
        )
        end_published_date: datetime = SchemaField(
            description="End date for published content",
        )
        include_text: List[str] = SchemaField(
            description="Text patterns to include (max 1 string, up to 5 words)",
            default=[],
            advanced=True,
        )
        exclude_text: List[str] = SchemaField(
            description="Text patterns to exclude (max 1 string, up to 5 words)",
            default=[],
            advanced=True,
        )
        contents: ContentSettings = SchemaField(
            description="Content retrieval settings",
            default=ContentSettings(),
            advanced=True,
        )

    class Output(BlockSchema):
        results: List[Any] = SchemaField(
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
            "numResults": input_data.number_of_results,
            "contents": input_data.contents.dict(),
        }

        optional_field_mapping = {
            "include_domains": "includeDomains",
            "exclude_domains": "excludeDomains",
            "include_text": "includeText",
            "exclude_text": "excludeText",
        }

        # Add optional fields if they have values
        for input_field, api_field in optional_field_mapping.items():
            value = getattr(input_data, input_field)
            if value:  # Only add non-empty values
                payload[api_field] = value

        date_field_mapping = {
            "start_crawl_date": "startCrawlDate",
            "end_crawl_date": "endCrawlDate",
            "start_published_date": "startPublishedDate",
            "end_published_date": "endPublishedDate",
        }

        # Add dates if they exist
        for input_field, api_field in date_field_mapping.items():
            value = getattr(input_data, input_field, None)
            if value:
                payload[api_field] = value.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            yield "results", data.get("results", [])
        except Exception as e:
            yield "error", str(e)
            yield "results", []
