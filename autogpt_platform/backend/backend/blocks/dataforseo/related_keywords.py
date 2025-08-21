"""
DataForSEO Google Related Keywords block.
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


class RelatedKeyword(BlockSchema):
    """Schema for a related keyword result."""

    keyword: str = SchemaField(description="The related keyword")
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
        description="SERP data for the keyword", default=None
    )
    clickstream_data: Optional[Dict[str, Any]] = SchemaField(
        description="Clickstream data metrics", default=None
    )


class DataForSeoRelatedKeywordsBlock(Block):
    """Block for getting related keywords from DataForSEO Labs."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = dataforseo.credentials_field(
            description="DataForSEO credentials (username and password)"
        )
        keywords: List[str] = SchemaField(
            description="Seed keywords to find related keywords for (up to 200)"
        )
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
        related_keywords: List[RelatedKeyword] = SchemaField(
            description="List of related keywords with metrics"
        )
        total_count: int = SchemaField(
            description="Total number of related keywords returned"
        )
        seed_keywords: List[str] = SchemaField(
            description="The seed keywords used for the query"
        )

    def __init__(self):
        super().__init__(
            id="8f2e4d6a-1b3c-4a5e-9d7f-2c8e6a4b3f1d",
            description="Get related keywords from DataForSEO Labs Google API",
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": dataforseo.get_test_credentials().model_dump(),
                "keywords": ["content marketing"],
                "location_code": 2840,
                "language_code": "en",
                "limit": 10,
            },
            test_credentials=dataforseo.get_test_credentials(),
            test_output=[
                ("related_keywords", lambda x: isinstance(x, list) and len(x) == 10),
                ("total_count", 10),
                ("seed_keywords", ["content marketing"]),
            ],
            test_mock={
                "_fetch_related_keywords": lambda *args, **kwargs: [
                    {
                        "items": [
                            {
                                "keyword": f"content strategy {i}",
                                "keyword_info": {
                                    "search_volume": 800 * (10 - i),
                                    "competition": 0.4 + (i * 0.06),
                                    "cpc": 3.0 + (i * 0.15),
                                },
                                "keyword_properties": {
                                    "keyword_difficulty": 45 + i,
                                },
                            }
                            for i in range(10)
                        ]
                    }
                ]
            },
        )

    async def _fetch_related_keywords(
        self,
        client: DataForSeoClient,
        input_data: Input,
    ) -> Any:
        """Private method to fetch related keywords - can be mocked for testing."""
        return await client.related_keywords(
            keywords=input_data.keywords,
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
        """Execute the related keywords query."""
        client = DataForSeoClient(credentials)

        results = await self._fetch_related_keywords(client, input_data)

        # Process and format the results
        related_keywords = []
        if results and len(results) > 0:
            # results is a list, get the first element
            first_result = results[0] if isinstance(results, list) else results
            items = (
                first_result.get("items", []) if isinstance(first_result, dict) else []
            )
            for item in items:
                # Create the RelatedKeyword object
                keyword = RelatedKeyword(
                    keyword=item.get("keyword", ""),
                    search_volume=item.get("keyword_info", {}).get("search_volume"),
                    competition=item.get("keyword_info", {}).get("competition"),
                    cpc=item.get("keyword_info", {}).get("cpc"),
                    keyword_difficulty=item.get("keyword_properties", {}).get(
                        "keyword_difficulty"
                    ),
                    serp_info=(
                        item.get("serp_info") if input_data.include_serp_info else None
                    ),
                    clickstream_data=(
                        item.get("clickstream_keyword_info")
                        if input_data.include_clickstream_data
                        else None
                    ),
                )
                related_keywords.append(keyword)

        yield "related_keywords", related_keywords
        yield "total_count", len(related_keywords)
        yield "seed_keywords", input_data.keywords
