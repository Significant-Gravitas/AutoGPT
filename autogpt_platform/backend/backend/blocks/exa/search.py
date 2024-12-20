from datetime import datetime
from typing import List

from backend.blocks.exa._auth import (
    ExaCredentials,
    ExaCredentialsField,
    ExaCredentialsInput,
)
from backend.blocks.exa.helpers import ContentSettings
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.request import requests


class ExaSearchBlock(Block):
    class Input(BlockSchema):
        credentials: ExaCredentialsInput = ExaCredentialsField()
        query: str = SchemaField(description="The search query")
        use_auto_prompt: bool = SchemaField(
            description="Whether to use autoprompt",
            default=True,
            advanced=True,
        )
        type: str = SchemaField(
            description="Type of search",
            default="",
            advanced=True,
        )
        category: str = SchemaField(
            description="Category to search within",
            default="",
            advanced=True,
        )
        number_of_results: int = SchemaField(
            description="Number of results to return",
            default=10,
            advanced=True,
        )
        include_domains: List[str] = SchemaField(
            description="Domains to include in search",
            default=[],
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
            description="Text patterns to include",
            default=[],
            advanced=True,
        )
        exclude_text: List[str] = SchemaField(
            description="Text patterns to exclude",
            default=[],
            advanced=True,
        )
        contents: ContentSettings = SchemaField(
            description="Content retrieval settings",
            default=ContentSettings(),
            advanced=True,
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
            "useAutoprompt": input_data.use_auto_prompt,
            "numResults": input_data.number_of_results,
            "contents": input_data.contents.dict(),
        }

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

        optional_field_mapping = {
            "type": "type",
            "category": "category",
            "include_domains": "includeDomains",
            "exclude_domains": "excludeDomains",
            "include_text": "includeText",
            "exclude_text": "excludeText",
        }

        # Add other fields
        for input_field, api_field in optional_field_mapping.items():
            value = getattr(input_data, input_field)
            if value:  # Only add non-empty values
                payload[api_field] = value

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            # Extract just the results array from the response
            yield "results", data.get("results", [])
        except Exception as e:
            yield "error", str(e)
            yield "results", []
