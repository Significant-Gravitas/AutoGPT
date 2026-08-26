from typing import Any

from pydantic import SecretStr

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


class SecurityHeadersAuditBlock(Block):
    """
    Grades a URL's HTTP security headers (HSTS, CSP, X-Frame-Options,
    X-Content-Type-Options, Referrer-Policy, Permissions-Policy) the same
    way a browser-facing security scanner would — useful for agents doing
    due diligence on a vendor/competitor site, or gating a deploy pipeline
    on a minimum security score.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = webmetadata_extractor.credentials_field(
            description="Web Metadata Extractor API key (free tier: 1,000 requests/month, no card required)."
        )
        url: str = SchemaField(
            description="The URL to audit", placeholder="https://example.com"
        )

    class Output(BlockSchemaOutput):
        security_score_percentage: float = SchemaField(
            description="Overall security-headers score, 0-100"
        )
        security_header_grades: dict[str, str] = SchemaField(
            description="Per-header grade: missing, weak, report-only, reasonable, or strong"
        )
        security_headers: dict[str, Any] = SchemaField(
            description="The raw header values that were graded"
        )

    def __init__(self):
        super().__init__(
            id="52d51ab9-1b71-40be-8dc5-673a0282d4fb",
            description="Grades a URL's HTTP security headers (HSTS, CSP, X-Frame-Options, etc.) with a 0-100 score.",
            categories={BlockCategory.SEARCH, BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "url": "https://example.com",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("security_score_percentage", 85.0),
                (
                    "security_header_grades",
                    {"strict_transport_security": "strong"},
                ),
                ("security_headers", dict),
            ],
            test_mock={
                "_fetch_security_audit": lambda *args, **kwargs: {
                    "security_score_percentage": 85.0,
                    "security_header_grades": {"strict_transport_security": "strong"},
                    "security_headers": {
                        "strict_transport_security": "max-age=31536000"
                    },
                }
            },
        )

    async def _fetch_security_audit(
        self, credentials: APIKeyCredentials, url: str
    ) -> dict:
        """Private method so the network call can be mocked in tests."""
        return await call_endpoint(credentials, "/api/v1/security", url)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self._fetch_security_audit(credentials, input_data.url)
        yield "security_score_percentage", data.get("security_score_percentage", 0)
        yield "security_header_grades", data.get("security_header_grades", {})
        yield "security_headers", data.get("security_headers", {})
