"""
DataForSEO Ranked Keywords block: the Google searches a site or page ranks for.
"""

from typing import Any

from backend.sdk import (
    BaseModel,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
    UserPasswordCredentials,
)
from backend.util.exceptions import BlockInputError

from ._api import DataForSeoClient, record_cost
from ._config import dataforseo

RANKED_KEYWORDS_PATH = "/v3/dataforseo_labs/google/ranked_keywords/live"
POSITION_FIELD = "ranked_serp_element.serp_item.rank_group"
# Highest estimated traffic first, so a small limit keeps the keywords that matter.
ORDER_BY = ["ranked_serp_element.serp_item.etv,desc"]


class RankedKeyword(BaseModel):
    keyword: str
    position: int | None
    url: str
    search_volume: int | None
    estimated_traffic: float | None
    cpc: float | None
    keyword_difficulty: int | None
    search_intent: str | None


class DataForSeoRankedKeywordsBlock(Block):
    """Keywords a domain or page ranks for in Google, with position and demand."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = dataforseo.credentials_field(
            description="DataForSEO credentials (username and password)"
        )
        target: str = SchemaField(
            description=(
                "Domain without https:// or www. (e.g. 'example.com') for the whole "
                "site, or a full page URL for one page. Works for any site, "
                "including competitors"
            )
        )
        location_code: int = SchemaField(
            description="Location code for targeting (e.g., 2840 for USA)",
            default=2840,
        )
        language_code: str = SchemaField(
            description="Language code (e.g., 'en' for English)", default="en"
        )
        min_position: int | None = SchemaField(
            description="Lowest position number to keep, e.g. 7 keeps positions 7, 8, 9 and on",
            default=None,
            ge=1,
            le=100,
        )
        max_position: int | None = SchemaField(
            description="Highest position number to keep, e.g. 15 keeps positions 1 to 15",
            default=None,
            ge=1,
            le=100,
        )
        limit: int = SchemaField(
            description="Maximum keywords to return, highest estimated traffic first",
            default=100,
            ge=1,
            le=1000,
        )

    class Output(BlockSchemaOutput):
        keywords: list[RankedKeyword] = SchemaField(
            description=(
                "Ranked keywords. position is the organic position (1 = first "
                "organic result); estimated_traffic is DataForSEO's monthly visit "
                "estimate for that position, not measured visits"
            )
        )
        total_count: int = SchemaField(
            description="How many keywords match in total, beyond the limit"
        )

    def __init__(self):
        super().__init__(
            id="8230e940-1049-4a32-a780-6c5d1ef14059",
            description=(
                "Get the Google keywords a domain or page ranks for from DataForSEO "
                "Labs, with position, ranking URL, search volume and estimated traffic"
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": dataforseo.get_test_credentials().model_dump(),
                "target": "example.com",
                "min_position": 7,
                "max_position": 15,
            },
            test_credentials=dataforseo.get_test_credentials(),
            test_output=[
                (
                    "keywords",
                    [
                        RankedKeyword(
                            keyword="soy candles",
                            position=8,
                            url="https://example.com/collections/soy",
                            search_volume=1900,
                            estimated_traffic=42.5,
                            cpc=1.2,
                            keyword_difficulty=31,
                            search_intent="commercial",
                        )
                    ],
                ),
                ("total_count", 57),
            ],
            test_mock={"_fetch": lambda *args, **kwargs: _TEST_RESULT},
            effect=BlockEffect.READ,
        )

    async def _fetch(self, client: DataForSeoClient, input_data: Input) -> list:
        return await client.post_task(RANKED_KEYWORDS_PATH, build_task(input_data))

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        if (
            input_data.min_position
            and input_data.max_position
            and input_data.min_position > input_data.max_position
        ):
            raise BlockInputError(
                message="min_position must not be greater than max_position",
                block_name=self.name,
                block_id=self.id,
            )
        client = DataForSeoClient(credentials)
        results = await self._fetch(client, input_data)
        record_cost(self, client.last_cost_usd)
        page = results[0] if results else {}
        yield "keywords", [parse_ranked_keyword(i) for i in page.get("items") or []]
        yield "total_count", page.get("total_count") or 0


def build_task(input_data: DataForSeoRankedKeywordsBlock.Input) -> dict[str, Any]:
    task: dict[str, Any] = {
        "target": input_data.target.strip(),
        "location_code": input_data.location_code,
        "language_code": input_data.language_code,
        "limit": input_data.limit,
        "order_by": ORDER_BY,
    }
    filters: list[Any] = []
    if input_data.min_position:
        filters.append([POSITION_FIELD, ">=", input_data.min_position])
    if input_data.max_position:
        filters.append([POSITION_FIELD, "<=", input_data.max_position])
    if filters:
        # DataForSEO joins conditions with a logical operator between them.
        task["filters"] = (
            filters[0] if len(filters) == 1 else [filters[0], "and", filters[1]]
        )
    return task


def parse_ranked_keyword(item: dict[str, Any]) -> RankedKeyword:
    data = item.get("keyword_data") or {}
    info = data.get("keyword_info") or {}
    serp_item = (item.get("ranked_serp_element") or {}).get("serp_item") or {}
    return RankedKeyword(
        keyword=data.get("keyword") or "",
        position=serp_item.get("rank_group"),
        url=serp_item.get("url") or "",
        search_volume=info.get("search_volume"),
        estimated_traffic=serp_item.get("etv"),
        cpc=info.get("cpc"),
        keyword_difficulty=(data.get("keyword_properties") or {}).get(
            "keyword_difficulty"
        ),
        search_intent=(data.get("search_intent_info") or {}).get("main_intent"),
    )


_TEST_RESULT = [
    {
        "total_count": 57,
        "items": [
            {
                "keyword_data": {
                    "keyword": "soy candles",
                    "keyword_info": {"search_volume": 1900, "cpc": 1.2},
                    "keyword_properties": {"keyword_difficulty": 31},
                    "search_intent_info": {"main_intent": "commercial"},
                },
                "ranked_serp_element": {
                    "serp_item": {
                        "rank_group": 8,
                        "url": "https://example.com/collections/soy",
                        "etv": 42.5,
                    }
                },
            }
        ],
    }
]
