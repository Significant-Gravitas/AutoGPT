from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.dataforb2b._api import DataForB2BClient
from backend.blocks.dataforb2b._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    DataForB2BCredentials,
    DataForB2BCredentialsField,
    DataForB2BCredentialsInput,
)
from backend.data.model import SchemaField

ENRICH_FLAGS = (
    "enrich_profile",
    "enrich_work_email",
    "enrich_personal_email",
    "enrich_github",
)


async def _enrich(payload: dict, credentials: DataForB2BCredentials) -> dict:
    client = DataForB2BClient(credentials)
    return await client.enrich_profile(payload)


class LinkedinProfileEnrichmentBlock(Block):
    """Enrich a LinkedIn profile with DataForB2B (full profile + email, phone, GitHub)."""

    class Input(BlockSchemaInput):
        profile_identifier: str = SchemaField(
            description="LinkedIn profile URL (or profile id) to enrich",
            advanced=False,
        )
        enrich_profile: bool = SchemaField(
            description="Return the full LinkedIn profile (role, experience, skills)",
            default=True,
            advanced=False,
        )
        enrich_work_email: bool = SchemaField(
            description="Find the professional / work email",
            default=False,
            advanced=False,
        )
        enrich_personal_email: bool = SchemaField(
            description="Find the personal email", default=False, advanced=False
        )
        enrich_github: bool = SchemaField(
            description="Find the GitHub profile", default=False, advanced=True
        )
        credentials: DataForB2BCredentialsInput = DataForB2BCredentialsField()

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Full enrichment response")
        error: str = SchemaField(
            description="Error message if enrichment failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="6b5eaff2-aded-47c4-9acc-e2047c76e72e",
            description=(
                "Look up and enrich a professional profile from a LinkedIn URL using "
                "DataForB2B's B2B database — returns the full profile (current role, "
                "experience, skills) plus work email, personal email and GitHub. Works "
                "as an email finder for lead enrichment, contact enrichment, cold "
                "outreach and CRM. Toggle the enrich_work_email flag to fetch only an "
                "email."
            ),
            categories={BlockCategory.SOCIAL, BlockCategory.DATA, BlockCategory.CRM},
            input_schema=LinkedinProfileEnrichmentBlock.Input,
            output_schema=LinkedinProfileEnrichmentBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "profile_identifier": "https://www.linkedin.com/in/johndoe",
                "enrich_profile": True,
                "enrich_work_email": True,
            },
            test_output=[
                (
                    "result",
                    {"profile": {"name": "John Doe"}, "work_email": "john@acme.com"},
                ),
            ],
            test_mock={
                "enrich_profile": lambda payload, credentials: {
                    "profile": {"name": "John Doe"},
                    "work_email": "john@acme.com",
                }
            },
        )

    @staticmethod
    async def enrich_profile(payload: dict, credentials: DataForB2BCredentials) -> dict:
        return await _enrich(payload, credentials)

    async def run(
        self, input_data: Input, *, credentials: DataForB2BCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.profile_identifier:
            raise ValueError("'profile_identifier' is required.")

        payload: dict = {"profile_identifier": input_data.profile_identifier}
        any_flag = False
        for flag in ENRICH_FLAGS:
            value = bool(getattr(input_data, flag, False))
            payload[flag] = value
            any_flag = any_flag or value
        if not any_flag:
            payload["enrich_profile"] = True

        yield "result", await self.enrich_profile(payload, credentials)


class CompanyEnrichmentBlock(Block):
    """Enrich a company with DataForB2B (firmographics from domain/name/LinkedIn URL)."""

    class Input(BlockSchemaInput):
        company_identifier: str = SchemaField(
            description="Company domain, name, or LinkedIn URL to enrich",
            advanced=False,
        )
        credentials: DataForB2BCredentialsInput = DataForB2BCredentialsField()

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Full company enrichment response")
        error: str = SchemaField(
            description="Error message if enrichment failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="32f80b6c-02e2-455f-af93-493514a19903",
            description=(
                "Look up and enrich a company using DataForB2B's B2B database — "
                "firmographics, headcount/size, industry, domain and social profiles "
                "from a company domain, name or LinkedIn URL. Account enrichment for "
                "B2B sales and CRM."
            ),
            categories={BlockCategory.SOCIAL, BlockCategory.DATA, BlockCategory.CRM},
            input_schema=CompanyEnrichmentBlock.Input,
            output_schema=CompanyEnrichmentBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "company_identifier": "google.com",
            },
            test_output=[
                ("result", {"name": "Google", "domain": "google.com"}),
            ],
            test_mock={
                "enrich_company": lambda company_identifier, credentials: {
                    "name": "Google",
                    "domain": "google.com",
                }
            },
        )

    @staticmethod
    async def enrich_company(
        company_identifier: str, credentials: DataForB2BCredentials
    ) -> dict:
        client = DataForB2BClient(credentials)
        return await client.enrich_company(company_identifier)

    async def run(
        self, input_data: Input, *, credentials: DataForB2BCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.company_identifier:
            raise ValueError("'company_identifier' is required.")
        yield "result", await self.enrich_company(
            input_data.company_identifier, credentials
        )
