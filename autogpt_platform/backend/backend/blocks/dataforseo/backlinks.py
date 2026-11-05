"""
DataForSEO Backlinks blocks: a site's link profile and the sites linking to it.
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

from ._api import DataForSeoClient, record_cost
from ._config import dataforseo

SUMMARY_PATH = "/v3/backlinks/summary/live"
REFERRING_DOMAINS_PATH = "/v3/backlinks/referring_domains/live"
TARGET_DESCRIPTION = (
    "Domain without https:// or www. (e.g. 'example.com'), a subdomain, or a full "
    "page URL. Works for any site, including competitors"
)
RANK_DESCRIPTION = (
    "DataForSEO's authority score, 0-1000 (higher is stronger); comparable to "
    "other tools' domain ratings but not the same scale"
)


class BacklinksSummary(BaseModel):
    rank: int | None
    backlinks: int | None
    referring_domains: int | None
    referring_domains_nofollow: int | None
    referring_main_domains: int | None
    broken_backlinks: int | None
    broken_pages: int | None
    spam_score: int | None
    first_seen: str | None


class ReferringDomain(BaseModel):
    domain: str
    rank: int | None
    backlinks: int | None
    spam_score: int | None
    first_seen: str | None


class DataForSeoBacklinksSummaryBlock(Block):
    """Headline backlink numbers for a domain or page."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = dataforseo.credentials_field(
            description="DataForSEO credentials (username and password)"
        )
        target: str = SchemaField(description=TARGET_DESCRIPTION)
        include_subdomains: bool = SchemaField(
            description="Count links to the target's subdomains too", default=True
        )

    class Output(BlockSchemaOutput):
        summary: BacklinksSummary = SchemaField(
            description=(
                f"Live backlinks and referring domains. rank: {RANK_DESCRIPTION}. "
                "broken_backlinks are links pointing at pages that no longer load; "
                "spam_score is 0-100"
            )
        )

    def __init__(self):
        super().__init__(
            id="15dfcc6b-f640-42b5-9258-bc36a771721a",
            description=(
                "Get a domain's or page's backlink profile from DataForSEO: authority "
                "rank, backlinks, referring domains, broken backlinks and spam score"
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": dataforseo.get_test_credentials().model_dump(),
                "target": "example.com",
            },
            test_credentials=dataforseo.get_test_credentials(),
            test_output=[
                (
                    "summary",
                    BacklinksSummary(
                        rank=374,
                        backlinks=24181,
                        referring_domains=2196,
                        referring_domains_nofollow=748,
                        referring_main_domains=1941,
                        broken_backlinks=73,
                        broken_pages=34,
                        spam_score=8,
                        first_seen="2023-04-01 00:00:00 +00:00",
                    ),
                )
            ],
            test_mock={"_fetch": lambda *args, **kwargs: [_TEST_SUMMARY]},
            effect=BlockEffect.READ,
        )

    async def _fetch(self, client: DataForSeoClient, input_data: Input) -> list:
        return await client.post_task(
            SUMMARY_PATH,
            {
                "target": input_data.target.strip(),
                "include_subdomains": input_data.include_subdomains,
            },
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        client = DataForSeoClient(credentials)
        results = await self._fetch(client, input_data)
        record_cost(self, client.last_cost_usd)
        data = results[0] if results else {}
        yield "summary", BacklinksSummary(
            rank=data.get("rank"),
            backlinks=data.get("backlinks"),
            referring_domains=data.get("referring_domains"),
            referring_domains_nofollow=data.get("referring_domains_nofollow"),
            referring_main_domains=data.get("referring_main_domains"),
            broken_backlinks=data.get("broken_backlinks"),
            broken_pages=data.get("broken_pages"),
            spam_score=data.get("backlinks_spam_score"),
            first_seen=data.get("first_seen"),
        )


class DataForSeoReferringDomainsBlock(Block):
    """The sites linking to a domain or page, strongest first."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = dataforseo.credentials_field(
            description="DataForSEO credentials (username and password)"
        )
        target: str = SchemaField(description=TARGET_DESCRIPTION)
        include_subdomains: bool = SchemaField(
            description="Count links to the target's subdomains too", default=True
        )
        limit: int = SchemaField(
            description="Maximum referring domains to return, highest rank first",
            default=100,
            ge=1,
            le=1000,
        )

    class Output(BlockSchemaOutput):
        referring_domains: list[ReferringDomain] = SchemaField(
            description=(
                "Sites with live links to the target, highest rank first. "
                f"rank: {RANK_DESCRIPTION}. backlinks is how many links that site "
                "has to the target; spam_score is 0-100"
            )
        )
        total_count: int = SchemaField(
            description="How many referring domains there are in total"
        )

    def __init__(self):
        super().__init__(
            id="5f530bbf-a98a-443b-9a99-a548f94c4ecb",
            description=(
                "List the sites linking to a domain or page from DataForSEO "
                "Backlinks, strongest first, with rank, link count and spam score"
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": dataforseo.get_test_credentials().model_dump(),
                "target": "example.com",
                "limit": 1,
            },
            test_credentials=dataforseo.get_test_credentials(),
            test_output=[
                (
                    "referring_domains",
                    [
                        ReferringDomain(
                            domain="docker.com",
                            rank=358,
                            backlinks=3,
                            spam_score=23,
                            first_seen="2026-01-14 14:55:16 +00:00",
                        )
                    ],
                ),
                ("total_count", 1941),
            ],
            test_mock={"_fetch": lambda *args, **kwargs: [_TEST_REFERRING]},
            effect=BlockEffect.READ,
        )

    async def _fetch(self, client: DataForSeoClient, input_data: Input) -> list:
        return await client.post_task(
            REFERRING_DOMAINS_PATH,
            {
                "target": input_data.target.strip(),
                "include_subdomains": input_data.include_subdomains,
                "exclude_internal_backlinks": True,
                "backlinks_status_type": "live",
                "limit": input_data.limit,
                "order_by": ["rank,desc"],
            },
        )

    async def run(
        self, input_data: Input, *, credentials: UserPasswordCredentials, **kwargs
    ) -> BlockOutput:
        client = DataForSeoClient(credentials)
        results = await self._fetch(client, input_data)
        record_cost(self, client.last_cost_usd)
        page = results[0] if results else {}
        yield "referring_domains", [
            _referring_domain(i) for i in page.get("items") or []
        ]
        yield "total_count", page.get("total_count") or 0


def _referring_domain(item: dict[str, Any]) -> ReferringDomain:
    return ReferringDomain(
        domain=item.get("domain") or "",
        rank=item.get("rank"),
        backlinks=item.get("backlinks"),
        spam_score=item.get("backlinks_spam_score"),
        first_seen=item.get("first_seen"),
    )


_TEST_SUMMARY = {
    "rank": 374,
    "backlinks": 24181,
    "referring_domains": 2196,
    "referring_domains_nofollow": 748,
    "referring_main_domains": 1941,
    "broken_backlinks": 73,
    "broken_pages": 34,
    "backlinks_spam_score": 8,
    "first_seen": "2023-04-01 00:00:00 +00:00",
}
_TEST_REFERRING = {
    "total_count": 1941,
    "items": [
        {
            "domain": "docker.com",
            "rank": 358,
            "backlinks": 3,
            "backlinks_spam_score": 23,
            "first_seen": "2026-01-14 14:55:16 +00:00",
        }
    ],
}
