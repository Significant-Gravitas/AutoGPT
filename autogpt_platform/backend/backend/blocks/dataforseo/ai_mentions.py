"""
DataForSEO AI Mentions block: where a site or term shows up in AI search answers.
"""

from typing import Any, Literal

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
from ._targets import normalise_domain

LLM_MENTIONS_PATH = "/v3/ai_optimization/llm_mentions/search/live"


class AiSource(BaseModel):
    url: str
    domain: str
    title: str


class AiMention(BaseModel):
    platform: str
    model: str
    question: str
    answer: str
    sources: list[AiSource]
    ai_search_volume: int | None
    last_seen: str | None


class DataForSeoAiMentionsBlock(Block):
    """AI answers (Google AI Overviews or ChatGPT) that cite a site or use a term."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = dataforseo.credentials_field(
            description="DataForSEO credentials (username and password)"
        )
        domain: str = SchemaField(
            description=(
                "Find answers that cite this site, without https:// or www. "
                "(e.g. 'example.com'). Leave empty to search by keyword only"
            ),
            default="",
        )
        keyword: str = SchemaField(
            description=(
                "Find answers whose question or text uses this term, e.g. a brand "
                "or product name. Leave empty to search by domain only"
            ),
            default="",
        )
        platform: Literal["google", "chat_gpt"] = SchemaField(
            description="google for AI Overviews, chat_gpt for ChatGPT (US English only)",
            default="google",
        )
        location_code: int = SchemaField(
            description="Location code for targeting (e.g., 2840 for USA)",
            default=2840,
        )
        language_code: str = SchemaField(
            description="Language code (e.g., 'en' for English)", default="en"
        )
        limit: int = SchemaField(
            description="Maximum answers to return, most-searched questions first",
            default=10,
            ge=1,
            le=100,
        )

    class Output(BlockSchemaOutput):
        mentions: list[AiMention] = SchemaField(
            description=(
                "AI answers matching the domain and/or keyword, with the question "
                "asked, the answer as markdown and the pages it cites. "
                "ai_search_volume is DataForSEO's monthly estimate for the question"
            )
        )
        total_count: int = SchemaField(
            description="How many matching answers there are in total"
        )

    def __init__(self):
        super().__init__(
            id="537353f0-4503-4682-be4c-5eaad8ed7235",
            description=(
                "Find AI search answers (Google AI Overviews or ChatGPT) that cite a "
                "website or mention a term, with the question, answer and cited "
                "sources, from DataForSEO AI Optimization"
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": dataforseo.get_test_credentials().model_dump(),
                "domain": "example.com",
                "limit": 1,
            },
            test_credentials=dataforseo.get_test_credentials(),
            test_output=[
                (
                    "mentions",
                    [
                        AiMention(
                            platform="google",
                            model="google_ai_overview",
                            question="best soy candles",
                            answer="Top picks include **Example**.",
                            sources=[
                                AiSource(
                                    url="https://example.com/soy",
                                    domain="example.com",
                                    title="Soy candles",
                                )
                            ],
                            ai_search_volume=880,
                            last_seen="2026-10-28 22:07:47 +00:00",
                        )
                    ],
                ),
                ("total_count", 12),
            ],
            test_mock={"_fetch": lambda *args, **kwargs: _TEST_RESULT},
            effect=BlockEffect.READ,
        )

    async def _fetch(self, client: DataForSeoClient, input_data: Input) -> list:
        return await client.post_task(LLM_MENTIONS_PATH, build_task(input_data))

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        if not normalise_domain(input_data.domain) and not input_data.keyword.strip():
            raise BlockInputError(
                message="Give a domain, a keyword, or both",
                block_name=self.name,
                block_id=self.id,
            )
        client = DataForSeoClient(credentials)
        results = await self._fetch(client, input_data)
        record_cost(self, client.last_cost_usd)
        page = results[0] if results else {}
        yield "mentions", [parse_mention(i) for i in page.get("items") or []]
        yield "total_count", page.get("total_count") or 0


def build_task(input_data: DataForSeoAiMentionsBlock.Input) -> dict[str, Any]:
    target: list[dict[str, str]] = []
    if domain := normalise_domain(input_data.domain):
        target.append({"domain": domain})
    if keyword := input_data.keyword.strip():
        target.append({"keyword": keyword})
    return {
        "target": target,
        "platform": input_data.platform,
        "location_code": input_data.location_code,
        "language_code": input_data.language_code,
        "limit": input_data.limit,
        "order_by": ["ai_search_volume,desc"],
    }


def parse_mention(item: dict[str, Any]) -> AiMention:
    return AiMention(
        platform=item.get("platform") or "",
        model=item.get("model_name") or "",
        question=item.get("question") or "",
        answer=item.get("answer") or "",
        sources=[
            AiSource(
                url=s.get("url") or "",
                domain=s.get("domain") or "",
                title=s.get("title") or "",
            )
            for s in item.get("sources") or []
        ],
        ai_search_volume=item.get("ai_search_volume"),
        last_seen=item.get("last_response_at"),
    )


_TEST_RESULT = [
    {
        "total_count": 12,
        "items": [
            {
                "platform": "google",
                "model_name": "google_ai_overview",
                "question": "best soy candles",
                "answer": "Top picks include **Example**.",
                "sources": [
                    {
                        "url": "https://example.com/soy",
                        "domain": "example.com",
                        "title": "Soy candles",
                        "position": 1,
                    }
                ],
                "ai_search_volume": 880,
                "last_response_at": "2026-10-28 22:07:47 +00:00",
            }
        ],
    }
]
