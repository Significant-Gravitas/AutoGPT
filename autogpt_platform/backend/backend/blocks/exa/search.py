from datetime import datetime
from typing import List, Optional
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
    text: Optional[dict] = SchemaField(
        description="Text content settings",
        default={"maxCharacters": 1000, "includeHtmlTags": False},
    )
    highlights: Optional[dict] = SchemaField(
        description="Highlight settings",
        default={"numSentences": 3, "highlightsPerUrl": 3},
    )
    summary: Optional[dict] = SchemaField(
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
        type: Optional[str] = SchemaField(
            description="Type of search",
            optional=True,
        )
        category: Optional[str] = SchemaField(
            description="Category to search within",
            optional=True,
        )
        numResults: int = SchemaField(
            description="Number of results to return",
            default=10,
        )
        includeDomains: Optional[List[str]] = SchemaField(
            description="Domains to include in search",
            optional=True,
        )
        excludeDomains: Optional[List[str]] = SchemaField(
            description="Domains to exclude from search",
            optional=True,
        )
        startCrawlDate: Optional[datetime] = SchemaField(
            description="Start date for crawled content",
            optional=True,
        )
        endCrawlDate: Optional[datetime] = SchemaField(
            description="End date for crawled content",
            optional=True,
        )
        startPublishedDate: Optional[datetime] = SchemaField(
            description="Start date for published content",
            optional=True,
        )
        endPublishedDate: Optional[datetime] = SchemaField(
            description="End date for published content",
            optional=True,
        )
        includeText: Optional[List[str]] = SchemaField(
            description="Text patterns to include",
            optional=True,
        )
        excludeText: Optional[List[str]] = SchemaField(
            description="Text patterns to exclude",
            optional=True,
        )
        contents: ContentSettings = SchemaField(
            description="Content retrieval settings",
            default=ContentSettings(),
        )

    class Output(BlockSchema):
        results: list = SchemaField(description="Search results from Exa")
        error: str = SchemaField(
            description="Error message if the search fails",
            optional=True,
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
        }

        optional_fields = [
            "type", "category", "includeDomains", "excludeDomains",
            "startCrawlDate", "endCrawlDate", "startPublishedDate",
            "endPublishedDate", "includeText", "excludeText",
        ]

        for field in optional_fields:
            value = getattr(input_data, field)
            if value is not None:
                if isinstance(value, datetime):
                    payload[field] = value.isoformat() + "Z"
                else:
                    payload[field] = value

        if input_data.contents:
            payload["contents"] = input_data.contents.dict(exclude_none=True)

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            results = response.json()
            yield "results", results
        except Exception as e:
            yield "error", str(e)
            yield "results", []