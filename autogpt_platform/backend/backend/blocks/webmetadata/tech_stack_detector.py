from typing import Any

from pydantic import BaseModel, SecretStr

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import call_endpoint
from ._config import webmetadata_extractor

TEST_CREDENTIALS = APIKeyCredentials(
    id="d6e9d3f5-8a1b-4c2e-9d7f-3b5a6c8e1f02",
    provider="web_metadata_extractor",
    api_key=SecretStr("mock-web-metadata-extractor-api-key"),
    title="Mock Web Metadata Extractor API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class TechnologyDetail(BaseModel):
    name: str
    category: str
    confidence: float
    evidence: list[str] = []


class TechStackDetectorBlock(Block):
    """
    Fingerprints the CMS/framework/analytics/e-commerce stack a URL is built
    on (WordPress, Shopify, Next.js, Cloudflare, Google Tag Manager, and
    40+ others) from response headers and page markers — useful for an
    agent qualifying leads by platform, or picking the right integration
    approach for a target site.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = webmetadata_extractor.credentials_field(
            description="Web Metadata Extractor API key (free tier: 1,000 requests/month, no card required)."
        )
        url: str = SchemaField(
            description="The URL to inspect", placeholder="https://example.com"
        )

    class Output(BlockSchemaOutput):
        detected_technologies: list[str] = SchemaField(
            description="Names of every technology detected"
        )
        technology: TechnologyDetail = SchemaField(
            description="A single detected technology, with category/confidence/evidence"
        )
        technologies: list[TechnologyDetail] = SchemaField(
            description="All detected technologies, with category/confidence/evidence"
        )

    def __init__(self):
        super().__init__(
            id="528a009b-4094-4292-b431-f257b52f9c29",
            description="Fingerprints the CMS/framework/analytics stack (WordPress, Shopify, Next.js, etc.) a URL is built on.",
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "url": "https://example.com",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("detected_technologies", ["WordPress"]),
                (
                    "technology",
                    TechnologyDetail(
                        name="WordPress", category="cms", confidence=0.9
                    ),
                ),
                (
                    "technologies",
                    [
                        TechnologyDetail(
                            name="WordPress", category="cms", confidence=0.9
                        )
                    ],
                ),
            ],
            test_mock={
                "_fetch_tech_stack": lambda *args, **kwargs: {
                    "detected_technologies": ["WordPress"],
                    "technology_details": [
                        {
                            "name": "WordPress",
                            "category": "cms",
                            "confidence": 0.9,
                            "evidence": ["wp-content"],
                        }
                    ],
                }
            },
        )

    async def _fetch_tech_stack(
        self, credentials: APIKeyCredentials, url: str
    ) -> dict[str, Any]:
        """Private method so the network call can be mocked in tests."""
        return await call_endpoint(credentials, "/api/v1/tech-stack", url)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self._fetch_tech_stack(credentials, input_data.url)

        yield "detected_technologies", data.get("detected_technologies", [])

        technologies = [
            TechnologyDetail(
                name=t.get("name", ""),
                category=t.get("category", ""),
                confidence=t.get("confidence", 0.0),
                evidence=t.get("evidence", []),
            )
            for t in data.get("technology_details", [])
        ]
        for technology in technologies:
            yield "technology", technology
        yield "technologies", technologies
