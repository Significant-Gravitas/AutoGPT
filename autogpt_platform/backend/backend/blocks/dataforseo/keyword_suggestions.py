"""
DataForSEO Google Keyword Suggestions block.
"""

from typing import Any, Dict, List, Optional

from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
    UserPasswordCredentials,
)

from ._api import DataForSeoClient
from ._config import dataforseo


class KeywordSuggestion(BlockSchema):
    """Schema for a keyword suggestion result."""

    keyword: str = SchemaField(description="The keyword suggestion")
    search_volume: Optional[int] = SchemaField(
        description="Monthly search volume", default=None
    )
    competition: Optional[float] = SchemaField(
        description="Competition level (0-1)", default=None
    )
    cpc: Optional[float] = SchemaField(
        description="Cost per click in USD", default=None
    )
    keyword_difficulty: Optional[int] = SchemaField(
        description="Keyword difficulty score", default=None
    )
    serp_info: Optional[Dict[str, Any]] = SchemaField(
        description="data from SERP for each keyword", default=None
    )
    clickstream_data: Optional[Dict[str, Any]] = SchemaField(
        description="Clickstream data metrics", default=None
    )


class DataForSeoKeywordSuggestionsBlock(Block):
    """Block for getting keyword suggestions from DataForSEO Labs."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = dataforseo.credentials_field(
            description="DataForSEO credentials (username and password)"
        )
        keyword: str = SchemaField(description="Seed keyword to get suggestions for")
        location_code: Optional[int] = SchemaField(
            description="Location code for targeting (e.g., 2840 for USA)",
            default=2840,  # USA
        )
        language_code: Optional[str] = SchemaField(
            description="Language code (e.g., 'en' for English)",
            default="en",
        )
        include_seed_keyword: bool = SchemaField(
            description="Include the seed keyword in results",
            default=True,
        )
        include_serp_info: bool = SchemaField(
            description="Include SERP information",
            default=False,
        )
        include_clickstream_data: bool = SchemaField(
            description="Include clickstream metrics",
            default=False,
        )
        limit: int = SchemaField(
            description="Maximum number of results (up to 3000)",
            default=100,
            ge=1,
            le=3000,
        )

    class Output(BlockSchema):
        suggestions: List[KeywordSuggestion] = SchemaField(
            description="List of keyword suggestions with metrics"
        )
        suggestion: KeywordSuggestion = SchemaField(
            description="A single keyword suggestion with metrics"
        )
        total_count: int = SchemaField(
            description="Total number of suggestions returned"
        )
        seed_keyword: str = SchemaField(
            description="The seed keyword used for the query"
        )
        error: str = SchemaField(description="Error message if the API call failed")

    def __init__(self):
        super().__init__(
            id="73c3e7c4-2b3f-4e9f-9e3e-8f7a5c3e2d45",
            description="Get keyword suggestions from DataForSEO Labs Google API",
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": dataforseo.get_test_credentials().model_dump(),
                "keyword": "digital marketing",
                "location_code": 2840,
                "language_code": "en",
                "limit": 1,
            },
            test_credentials=dataforseo.get_test_credentials(),
            test_output=[
                (
                    "suggestion",
                    lambda x: hasattr(x, "keyword")
                    and x.keyword == "digital marketing strategy",
                ),
                ("suggestions", lambda x: isinstance(x, list) and len(x) == 1),
                ("total_count", 1),
                ("seed_keyword", "digital marketing"),
            ],
            test_mock={
                "_fetch_keyword_suggestions": lambda *args, **kwargs: [
                    {
                        "items": [
                            {
                                "keyword": "digital marketing strategy",
                                "keyword_info": {
                                    "search_volume": 10000,
                                    "competition": 0.5,
                                    "cpc": 2.5,
                                },
                                "keyword_properties": {
                                    "keyword_difficulty": 50,
                                },
                            }
                        ]
                    }
                ]
            },
        )

    async def _fetch_keyword_suggestions(
        self,
        client: DataForSeoClient,
        input_data: Input,
    ) -> Any:
        """Private method to fetch keyword suggestions - can be mocked for testing."""
        return await client.keyword_suggestions(
            keyword=input_data.keyword,
            location_code=input_data.location_code,
            language_code=input_data.language_code,
            include_seed_keyword=input_data.include_seed_keyword,
            include_serp_info=input_data.include_serp_info,
            include_clickstream_data=input_data.include_clickstream_data,
            limit=input_data.limit,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: UserPasswordCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the keyword suggestions query."""
        try:
            client = DataForSeoClient(credentials)

            results = await self._fetch_keyword_suggestions(client, input_data)

            # Process and format the results
            suggestions = []
            if results and len(results) > 0:
                # results is a list, get the first element
                first_result = results[0] if isinstance(results, list) else results
                items = (
                    first_result.get("items", [])
                    if isinstance(first_result, dict)
                    else []
                )
                if items is None:
                    items = []
                for item in items:
                    # Create the KeywordSuggestion object
                    suggestion = KeywordSuggestion(
                        keyword=item.get("keyword", ""),
                        search_volume=item.get("keyword_info", {}).get("search_volume"),
                        competition=item.get("keyword_info", {}).get("competition"),
                        cpc=item.get("keyword_info", {}).get("cpc"),
                        keyword_difficulty=item.get("keyword_properties", {}).get(
                            "keyword_difficulty"
                        ),
                        serp_info=(
                            item.get("serp_info")
                            if input_data.include_serp_info
                            else None
                        ),
                        clickstream_data=(
                            item.get("clickstream_keyword_info")
                            if input_data.include_clickstream_data
                            else None
                        ),
                    )
                    yield "suggestion", suggestion
                    suggestions.append(suggestion)

            yield "suggestions", suggestions
            yield "total_count", len(suggestions)
            yield "seed_keyword", input_data.keyword
        except Exception as e:
            yield "error", f"Failed to fetch keyword suggestions: {str(e)}"


class KeywordSuggestionExtractorBlock(Block):
    """Extracts individual fields from a KeywordSuggestion object."""

    class Input(BlockSchema):
        suggestion: KeywordSuggestion = SchemaField(
            description="The keyword suggestion object to extract fields from"
        )

    class Output(BlockSchema):
        keyword: str = SchemaField(description="The keyword suggestion")
        search_volume: Optional[int] = SchemaField(
            description="Monthly search volume", default=None
        )
        competition: Optional[float] = SchemaField(
            description="Competition level (0-1)", default=None
        )
        cpc: Optional[float] = SchemaField(
            description="Cost per click in USD", default=None
        )
        keyword_difficulty: Optional[int] = SchemaField(
            description="Keyword difficulty score", default=None
        )
        serp_info: Optional[Dict[str, Any]] = SchemaField(
            description="data from SERP for each keyword", default=None
        )
        clickstream_data: Optional[Dict[str, Any]] = SchemaField(
            description="Clickstream data metrics", default=None
        )

    def __init__(self):
        super().__init__(
            id="4193cb94-677c-48b0-9eec-6ac72fffd0f2",
            description="Extract individual fields from a KeywordSuggestion object",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "suggestion": KeywordSuggestion(
                    keyword="test keyword",
                    search_volume=1000,
                    competition=0.5,
                    cpc=2.5,
                    keyword_difficulty=60,
                ).model_dump()
            },
            test_output=[
                ("keyword", "test keyword"),
                ("search_volume", 1000),
                ("competition", 0.5),
                ("cpc", 2.5),
                ("keyword_difficulty", 60),
                ("serp_info", None),
                ("clickstream_data", None),
            ],
        )

    async def run(
        self,
        input_data: Input,
        **kwargs,
    ) -> BlockOutput:
        """Extract fields from the KeywordSuggestion object."""
        suggestion = input_data.suggestion

        yield "keyword", suggestion.keyword
        yield "search_volume", suggestion.search_volume
        yield "competition", suggestion.competition
        yield "cpc", suggestion.cpc
        yield "keyword_difficulty", suggestion.keyword_difficulty
        yield "serp_info", suggestion.serp_info
        yield "clickstream_data", suggestion.clickstream_data
