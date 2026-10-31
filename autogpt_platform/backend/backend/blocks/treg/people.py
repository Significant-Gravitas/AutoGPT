"""Treg blocks that find people and look up a person."""

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
    typed_fields,
)
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import mock_routed, routed_details

LIMIT_DESCRIPTION = (
    "How many results to return. Most providers bill per result, so this is "
    "the main price dial."
)

PERSON_FIELDS: dict[str, type] = {
    "full_name": str,
    "first_name": str,
    "last_name": str,
    "title": str,
    "company": str,
    "company_domain": str,
    "linkedin_url": str,
    "location": str,
}

TEST_PERSON = {
    "full_name": "Jane Doe",
    "first_name": "Jane",
    "last_name": "Doe",
    "title": "VP Platform",
    "company": "Outerbounds",
    "company_domain": "outerbounds.com",
    "linkedin_url": "https://www.linkedin.com/in/janedoe",
    "location": "San Francisco, California",
}


class TregSearchPeopleBlock(TregBlockBase):
    class Input(TregRoutedInput):
        description: str = SchemaField(
            default="",
            description="Who you are looking for, in plain words, e.g. 'heads of "
            "platform engineering at Series B AI infrastructure startups'.",
        )
        company_domain: str = SchemaField(
            default="",
            description="Only people at this company, by website domain, e.g. "
            "'modal.com'.",
        )
        job_title: str = SchemaField(
            default="", description="Only people with this job title, e.g. 'CTO'."
        )
        full_name: str = SchemaField(
            default="", description="Look for a person by name."
        )
        keywords: list[str] = SchemaField(
            default_factory=list,
            description="Skills or topics they must match, e.g. ['kubernetes'].",
        )
        country: str = SchemaField(
            default="",
            advanced=True,
            description="Two-letter country code, e.g. 'US', where the provider "
            "supports it.",
        )
        location: str = SchemaField(
            default="",
            advanced=True,
            description="A place, e.g. 'London, United Kingdom', where the "
            "provider supports it.",
        )
        limit: int = SchemaField(
            default=10, ge=1, le=100, description=LIMIT_DESCRIPTION
        )

    class Output(TregRoutedOutput):
        people: list[dict[str, Any]] = SchemaField(
            description="Matching people, in the serving provider's own format."
        )
        count: int = SchemaField(description="How many people came back.")
        ignored_filters: list[str] = SchemaField(
            description="Filters the serving provider could not apply, so the "
            "results may be broader than asked. Emitted only when there are some."
        )

    def __init__(self):
        super().__init__(
            id="5a0e3c9d-7b14-4f2e-86a1-c4d9e2b7f051",
            description=(
                "Find people by description, company, job title or name. Treg routes "
                "the search across many people-data providers, cheapest first."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "company_domain": "outerbounds.com",
                "job_title": "VP Platform",
                "limit": 5,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("people", [TEST_PERSON]),
                ("count", 1),
                *routed_details(served_by="aiark.people.search", cost_usd=0.0026),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {"people": [TEST_PERSON], "count": 1},
                    served_by="aiark.people.search",
                    cost_usd=0.0026,
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
                "company": bool(input_data.company_domain),
                "title": bool(input_data.job_title),
                "name": bool(input_data.full_name),
            },
            "give a description, a company domain, a job title or a name.",
        )
        routed = await self.route(
            credentials,
            "people.search",
            {
                "q": input_data.description,
                "company_domain": input_data.company_domain,
                "title": input_data.job_title,
                "full_name": input_data.full_name,
                "keywords": input_data.keywords,
                "country": input_data.country,
                "location": input_data.location,
                "limit": input_data.limit,
            },
            input_data,
        )
        people = [p for p in routed.get("people") or [] if isinstance(p, dict)]
        yield "people", people
        yield "count", len(people)
        if routed.meta.ignored_filters:
            yield "ignored_filters", routed.meta.ignored_filters
        for output in self.route_details(routed, bool(people)):
            yield output


class TregEnrichPersonBlock(TregBlockBase):
    class Input(TregRoutedInput):
        email: str = SchemaField(default="", description="The person's email.")
        linkedin_url: str = SchemaField(
            default="", description="Their LinkedIn profile URL."
        )
        full_name: str = SchemaField(
            default="", description="Their full name, with a company domain."
        )
        company_domain: str = SchemaField(
            default="", description="Their company's website domain."
        )

    class Output(TregRoutedOutput):
        full_name: str = SchemaField(description="Full name.")
        first_name: str = SchemaField(description="First name.")
        last_name: str = SchemaField(description="Last name.")
        title: str = SchemaField(description="Current job title.")
        company: str = SchemaField(description="Current company.")
        company_domain: str = SchemaField(description="Current company's domain.")
        linkedin_url: str = SchemaField(description="LinkedIn profile URL.")
        location: str = SchemaField(description="Where they are based.")

    def __init__(self):
        super().__init__(
            id="c3d81f5e-2a9b-4e70-b6c4-8f1e5a3d9c27",
            description=(
                "Look up a person (name, job title, company, LinkedIn and "
                "location) from an email, a LinkedIn URL, or a name and company "
                "domain, through Treg."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "email": "jane@outerbounds.com",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                *TEST_PERSON.items(),
                *routed_details(served_by="pdl.person.enrich", cost_usd=0.01),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    TEST_PERSON, served_by="pdl.person.enrich", cost_usd=0.01
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        require_one(
            {
                "email": bool(input_data.email),
                "linkedin": bool(input_data.linkedin_url),
                "name": bool(input_data.full_name and input_data.company_domain),
            },
            "give an email, a LinkedIn URL, or a full name and company domain.",
        )
        routed = await self.route(
            credentials,
            "people.enrich",
            {
                "email": input_data.email,
                "linkedin_url": input_data.linkedin_url,
                "full_name": input_data.full_name,
                "domain": input_data.company_domain,
            },
            input_data,
        )
        found = bool(routed.get("full_name"))
        if found:
            for output in typed_fields(routed.output, PERSON_FIELDS):
                yield output
        for output in self.route_details(routed, found):
            yield output


class TregFindLinkedInFromEmailBlock(TregBlockBase):
    class Input(TregRoutedInput):
        email: str = SchemaField(
            description="A work email address, e.g. 'jane@outerbounds.com'.",
            min_length=3,
        )

    class Output(TregRoutedOutput):
        linkedin_url: str = SchemaField(description="Their LinkedIn profile URL.")
        full_name: str = SchemaField(description="Their full name, when known.")

    def __init__(self):
        super().__init__(
            id="0f6a2d8c-9e3b-4a17-85d2-b1c7e4f9a362",
            description="Find the LinkedIn profile behind a work email, through Treg.",
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "email": "jane@outerbounds.com",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("linkedin_url", TEST_PERSON["linkedin_url"]),
                ("full_name", "Jane Doe"),
                *routed_details(served_by="blitzapi.email.to.linkedin"),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {
                        "linkedin_url": TEST_PERSON["linkedin_url"],
                        "full_name": "Jane Doe",
                    },
                    served_by="blitzapi.email.to.linkedin",
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        routed = await self.route(
            credentials,
            "people.identity.resolve",
            {"email": input_data.email},
            input_data,
        )
        found = bool(routed.get("linkedin_url"))
        if found:
            for output in typed_fields(
                routed.output, {"linkedin_url": str, "full_name": str}
            ):
                yield output
        for output in self.route_details(routed, found):
            yield output
