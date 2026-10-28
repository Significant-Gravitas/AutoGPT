"""Treg blocks that find companies and look them up."""

from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    SchemaField,
)

from ._base import (
    TregBlockBase,
    TregRoutedInput,
    TregRoutedOutput,
    require_one,
    rows,
    typed_fields,
)
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import mock_routed, routed_details
from .people import LIMIT_DESCRIPTION

COMPANY_FIELDS: dict[str, type] = {
    "name": str,
    "domain": str,
    "website": str,
    "description": str,
    "industry": str,
    "employees": str,
    "founded": int,
    "location": str,
    "linkedin_url": str,
}

TEST_COMPANY = {
    "name": "Modal",
    "domain": "modal.com",
    "website": "https://modal.com",
    "description": "Serverless cloud for AI and data teams.",
    "industry": "Software Development",
    "employees": "51-200",
    "founded": 2021,
    "location": "New York, United States",
    "linkedin_url": "https://www.linkedin.com/company/modal-labs",
}
TEST_ROWS = [{"name": "Baseten", "domain": "baseten.co"}]


class _CompanyListOutput(TregRoutedOutput):
    companies: list[dict[str, Any]] = SchemaField(
        description="Matching companies, in the serving provider's own format."
    )
    count: int = SchemaField(description="How many companies came back.")


class TregSearchCompaniesBlock(TregBlockBase):
    class Input(TregRoutedInput):
        description: str = SchemaField(
            default="",
            description="What kind of company, in plain words, e.g. 'Series A "
            "developer tools startups in Berlin'.",
        )
        name: str = SchemaField(default="", description="Look for a company by name.")
        industry: str = SchemaField(
            default="", description="Only companies in this industry."
        )
        technology: str = SchemaField(
            default="",
            description="Only companies that use this technology, e.g. 'Snowflake'.",
        )
        country: str = SchemaField(
            default="",
            advanced=True,
            description="Two-letter country code, e.g. 'DE', where the provider "
            "supports it.",
        )
        limit: int = SchemaField(
            default=10, ge=1, le=100, description=LIMIT_DESCRIPTION
        )

    class Output(_CompanyListOutput):
        pass

    def __init__(self):
        super().__init__(
            id="2b8f6e1a-4c3d-4a95-b7e0-5d1c9a8f3e46",
            description=(
                "Find companies by description, name, industry or the technology "
                "they use. Treg routes the search across 24 company-data providers."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "description": "GPU inference platforms",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("companies", TEST_ROWS),
                ("count", 1),
                *routed_details(served_by="thecompaniesapi.search", cost_usd=0.0019),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {"companies": TEST_ROWS, "count": 1},
                    served_by="thecompaniesapi.search",
                    cost_usd=0.0019,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        require_one(
            {
                "description": bool(input_data.description),
                "name": bool(input_data.name),
                "industry": bool(input_data.industry),
                "technology": bool(input_data.technology),
            },
            "give a description, a name, an industry or a technology.",
        )
        routed = await self.route(
            credentials,
            "companies.search",
            {
                "q": input_data.description,
                "name": input_data.name,
                "industry": input_data.industry,
                "technology": input_data.technology,
                "country": input_data.country,
                "limit": input_data.limit,
            },
            input_data,
        )
        companies = rows(routed, "companies")
        yield "companies", companies
        yield "count", len(companies)
        for output in self.route_details(routed, bool(companies)):
            yield output


class TregEnrichCompanyBlock(TregBlockBase):
    class Input(TregRoutedInput):
        domain: str = SchemaField(
            default="", description="The company's website domain, e.g. 'modal.com'."
        )
        name: str = SchemaField(default="", description="The company's name.")
        linkedin_url: str = SchemaField(
            default="", description="Its LinkedIn company page URL."
        )
        email: str = SchemaField(
            default="", description="Anyone's work email at the company."
        )

    class Output(TregRoutedOutput):
        name: str = SchemaField(description="Company name.")
        domain: str = SchemaField(description="Website domain.")
        website: str = SchemaField(description="Website URL.")
        description: str = SchemaField(description="What the company does.")
        industry: str = SchemaField(description="Industry.")
        employees: str = SchemaField(description="Headcount or headcount range.")
        founded: int = SchemaField(description="Year founded.")
        location: str = SchemaField(description="Headquarters.")
        linkedin_url: str = SchemaField(description="LinkedIn company page URL.")

    def __init__(self):
        super().__init__(
            id="8c5a3f7d-0e2b-4d61-9a4c-6b8e1f3d7a25",
            description=(
                "Look up a company (what it does, size, industry, founding year, "
                "headquarters and LinkedIn) from its domain, name, LinkedIn page "
                "or an employee's email, through Treg."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "domain": "modal.com"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                *TEST_COMPANY.items(),
                *routed_details(served_by="branddev.company.enrich", cost_usd=0.0005),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    TEST_COMPANY, served_by="branddev.company.enrich", cost_usd=0.0005
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        require_one(
            {
                "domain": bool(input_data.domain),
                "name": bool(input_data.name),
                "linkedin": bool(input_data.linkedin_url),
                "email": bool(input_data.email),
            },
            "give a domain, a name, a LinkedIn URL or an email.",
        )
        routed = await self.route(
            credentials,
            "companies.enrich",
            {
                "domain": input_data.domain,
                "name": input_data.name,
                "linkedin_url": input_data.linkedin_url,
                "email": input_data.email,
            },
            input_data,
        )
        found = bool(routed.get("name"))
        if found:
            for output in typed_fields(routed.output, COMPANY_FIELDS):
                yield output
        for output in self.route_details(routed, found):
            yield output


class TregFindSimilarCompaniesBlock(TregBlockBase):
    class Input(TregRoutedInput):
        domain: str = SchemaField(
            description="The company to find lookalikes of, by website domain, "
            "e.g. 'modal.com'.",
            min_length=1,
        )

    class Output(_CompanyListOutput):
        pass

    def __init__(self):
        super().__init__(
            id="d4e9b2c6-5f8a-4173-a0d3-2c7b6e9f1a58",
            description="Find companies similar to one you name, through Treg.",
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "domain": "modal.com"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("companies", TEST_ROWS),
                ("count", 1),
                *routed_details(served_by="companyenrich.similar", cost_usd=0.0089),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {"companies": TEST_ROWS, "count": 1},
                    served_by="companyenrich.similar",
                    cost_usd=0.0089,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        routed = await self.route(
            credentials, "companies.similar", {"domain": input_data.domain}, input_data
        )
        companies = rows(routed, "companies")
        yield "companies", companies
        yield "count", len(companies)
        for output in self.route_details(routed, bool(companies)):
            yield output


class TregFindCompanyJobsBlock(TregBlockBase):
    class Input(TregRoutedInput):
        domain: str = SchemaField(
            default="", description="The company's website domain."
        )
        name: str = SchemaField(default="", description="The company's name.")
        limit: int = SchemaField(
            default=10, ge=1, le=100, description=LIMIT_DESCRIPTION
        )

    class Output(TregRoutedOutput):
        jobs: list[dict[str, Any]] = SchemaField(
            description="Open job listings, in the serving provider's own format."
        )
        count: int = SchemaField(description="How many listings came back.")

    def __init__(self):
        super().__init__(
            id="6f1c8a3e-b7d2-4e09-95a6-3e8d0c4b7f12",
            description=(
                "List a company's open job listings, a strong signal of what it is "
                "investing in, through Treg."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "domain": "modal.com"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("jobs", [{"title": "Platform Engineer"}]),
                ("count", 1),
                *routed_details(served_by="coresignal.jobs.search", cost_usd=0.0),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {"jobs": [{"title": "Platform Engineer"}], "count": 1},
                    served_by="coresignal.jobs.search",
                    cost_usd=0.0,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        require_one(
            {"domain": bool(input_data.domain), "name": bool(input_data.name)},
            "give the company's domain or name.",
        )
        routed = await self.route(
            credentials,
            "companies.jobs.search",
            {
                "domain": input_data.domain,
                "name": input_data.name,
                "limit": input_data.limit,
            },
            input_data,
        )
        jobs = rows(routed, "jobs")
        yield "jobs", jobs
        yield "count", len(jobs)
        for output in self.route_details(routed, bool(jobs)):
            yield output
