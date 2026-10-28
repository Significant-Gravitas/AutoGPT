"""Treg blocks that search the web, Google, YouTube and Google Maps."""

from enum import Enum
from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    SchemaField,
)

from ._base import TregBlockBase, TregRoutedInput, TregRoutedOutput, rows
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import mock_routed, routed_details


class TregSearchSource(str, Enum):
    WEB = "web"
    NEWS = "news"
    GOOGLE = "google"
    GOOGLE_NEWS = "google_news"
    RESEARCH_PAPERS = "research_papers"
    YOUTUBE = "youtube"


# Each source's routed tool, the list its answer carries, and whether it takes
# a country and language.
SEARCH_SOURCES: dict[TregSearchSource, tuple[str, str, bool]] = {
    TregSearchSource.WEB: ("web.search", "results", False),
    TregSearchSource.NEWS: ("web.search.news", "results", False),
    TregSearchSource.GOOGLE: ("google.serp.organic", "results", True),
    TregSearchSource.GOOGLE_NEWS: ("google.serp.news", "results", True),
    TregSearchSource.RESEARCH_PAPERS: ("web.search.publications", "results", False),
    TregSearchSource.YOUTUBE: ("youtube.search.videos", "videos", False),
}

TEST_RESULTS = [{"title": "Modal", "url": "https://modal.com"}]


class TregWebSearchBlock(TregBlockBase):
    class Input(TregRoutedInput):
        query: str = SchemaField(
            description="What to search for, e.g. 'serverless GPU pricing'.",
            min_length=1,
        )
        source: TregSearchSource = SchemaField(
            default=TregSearchSource.WEB,
            description="Where to search: the open web, recent news, Google, "
            "Google News, research papers or YouTube.",
        )
        limit: int = SchemaField(
            default=10,
            ge=1,
            le=100,
            description="How many results to return, where the provider accepts a "
            "limit (YouTube does not).",
        )
        country: str = SchemaField(
            default="",
            advanced=True,
            description="Two-letter country code for Google and Google News, "
            "e.g. 'gb'. Defaults to 'us'.",
        )
        language: str = SchemaField(
            default="",
            advanced=True,
            description="Language code for Google and Google News, e.g. 'de'. "
            "Defaults to 'en'.",
        )

    class Output(TregRoutedOutput):
        results: list[dict[str, Any]] = SchemaField(
            description="Results in ranked order, in the serving provider's own "
            "format; most have a title and a URL."
        )
        count: int = SchemaField(description="How many results came back.")

    def __init__(self):
        super().__init__(
            id="f2a7c4e9-3b8d-4f16-a0c5-7e9b1d3f6a84",
            description=(
                "Search the web, news, Google, research papers or YouTube through "
                "Treg, which picks the cheapest working search provider."
            ),
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "serverless GPU platforms",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("results", TEST_RESULTS),
                ("count", 1),
                *routed_details(served_by="exa.search", cost_usd=0.005),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {"results": TEST_RESULTS, "count": 1},
                    served_by="exa.search",
                    cost_usd=0.005,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        capability, field, localised = SEARCH_SOURCES[input_data.source]
        body: dict[str, Any] = {"q": input_data.query}
        if input_data.source != TregSearchSource.YOUTUBE:
            body["limit"] = input_data.limit
        if localised:
            body["country"] = input_data.country
            body["language"] = input_data.language
        routed = await self.route(credentials, capability, body, input_data)
        results = rows(routed, field)
        yield "results", results
        yield "count", len(results)
        for output in self.route_details(routed, bool(results)):
            yield output


class TregGoogleMapsSearchBlock(TregBlockBase):
    class Input(TregRoutedInput):
        query: str = SchemaField(
            description="What and where, e.g. 'coffee roasters in Bristol'.",
            min_length=1,
        )
        country: str = SchemaField(
            default="",
            advanced=True,
            description="Two-letter country code, e.g. 'gb'. Defaults to 'us'.",
        )
        language: str = SchemaField(
            default="", advanced=True, description="Language code. Defaults to 'en'."
        )

    class Output(TregRoutedOutput):
        places: list[dict[str, Any]] = SchemaField(
            description="Matching places, in the serving provider's own format; "
            "most have a name, address, rating and place id."
        )
        count: int = SchemaField(description="How many places came back.")

    def __init__(self):
        super().__init__(
            id="1d9e6b3a-8f2c-4e57-b9a1-4c6f0e8d2b75",
            description="Find businesses and places on Google Maps through Treg.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "coffee roasters in Bristol",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("places", [{"name": "Extract Coffee Roasters"}]),
                ("count", 1),
                *routed_details(served_by="serper.maps", cost_usd=0.001),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {"places": [{"name": "Extract Coffee Roasters"}], "count": 1},
                    served_by="serper.maps",
                    cost_usd=0.001,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        routed = await self.route(
            credentials,
            "google.serp.maps",
            {
                "q": input_data.query,
                "country": input_data.country,
                "language": input_data.language,
            },
            input_data,
        )
        places = rows(routed, "places")
        yield "places", places
        yield "count", len(places)
        for output in self.route_details(routed, bool(places)):
            yield output
