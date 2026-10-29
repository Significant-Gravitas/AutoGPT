"""Treg blocks that read public LinkedIn profiles and company pages."""

from urllib.parse import urlparse

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    SchemaField,
)

from ._base import TregBlockBase, TregRoutedInput, TregRoutedOutput, typed_fields
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import mock_routed, routed_details
from .people import TEST_PERSON

PROFILE_FIELDS: dict[str, type] = {
    "full_name": str,
    "first_name": str,
    "last_name": str,
    "headline": str,
    "location": str,
    "about": str,
    "followers": int,
    "connections": int,
    "linkedin_url": str,
}


class TregGetLinkedInProfileBlock(TregBlockBase):
    class Input(TregRoutedInput):
        linkedin_url: str = SchemaField(
            description="The profile URL, e.g. 'https://www.linkedin.com/in/janedoe', "
            "or just the handle, e.g. 'janedoe'.",
            min_length=1,
        )

    class Output(TregRoutedOutput):
        full_name: str = SchemaField(description="Full name.")
        first_name: str = SchemaField(description="First name.")
        last_name: str = SchemaField(description="Last name.")
        headline: str = SchemaField(description="Profile headline.")
        location: str = SchemaField(description="Location on the profile.")
        about: str = SchemaField(description="The About section.")
        followers: int = SchemaField(description="Follower count.")
        connections: int = SchemaField(description="Connection count.")
        linkedin_url: str = SchemaField(description="Canonical profile URL.")

    def __init__(self):
        super().__init__(
            id="7e4c0b9a-1d6f-4b38-a2e5-9c3f7d1b5a80",
            description=(
                "Read a LinkedIn member's public profile through Treg, without a "
                "LinkedIn account or scraper."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "linkedin_url": "janedoe",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("full_name", "Jane Doe"),
                ("headline", "VP Platform at Outerbounds"),
                ("followers", 1200),
                ("linkedin_url", TEST_PERSON["linkedin_url"]),
                *routed_details(served_by="enrichlayer.linkedin.profile"),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {
                        "full_name": "Jane Doe",
                        "headline": "VP Platform at Outerbounds",
                        "followers": "1,200",
                        "connections": "500+",
                        "linkedin_url": TEST_PERSON["linkedin_url"],
                    },
                    served_by="enrichlayer.linkedin.profile",
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        routed = await self.route(
            credentials,
            "linkedin.user.profile",
            linkedin_reference(input_data.linkedin_url),
            input_data,
        )
        found = bool(routed.get("full_name"))
        if found:
            for output in typed_fields(routed.output, PROFILE_FIELDS):
                yield output
        for output in self.route_details(routed, found):
            yield output


LINKEDIN_COMPANY_FIELDS: dict[str, type] = {
    "name": str,
    "description": str,
    "website": str,
    "followers": int,
    "employees": str,
    "location": str,
    "linkedin_url": str,
}
TEST_COMPANY_PAGE = {
    "name": "Modal",
    "description": "Serverless cloud for AI and data teams.",
    "website": "https://modal.com",
    "followers": 21000,
    "employees": "51-200",
    "location": "New York, NY",
    "linkedin_url": "https://www.linkedin.com/company/modal-labs",
}


def linkedin_reference(reference: str) -> dict[str, str]:
    """A LinkedIn URL is sent as a URL, anything else as a handle."""
    reference = reference.strip()
    candidate = reference if "://" in reference else f"https://{reference}"
    host = (urlparse(candidate).hostname or "").lower()
    if host == "linkedin.com" or host.endswith(".linkedin.com"):
        return {"linkedin_url": reference}
    return {"linkedin_handle": reference}


class TregGetLinkedInCompanyBlock(TregBlockBase):
    class Input(TregRoutedInput):
        linkedin_url: str = SchemaField(
            description="The company page URL, e.g. "
            "'https://www.linkedin.com/company/modal-labs', or just the handle, "
            "e.g. 'modal-labs'.",
            min_length=1,
        )

    class Output(TregRoutedOutput):
        name: str = SchemaField(description="Company name.")
        description: str = SchemaField(description="The page's About text.")
        website: str = SchemaField(description="Website URL.")
        followers: int = SchemaField(description="Follower count.")
        employees: str = SchemaField(description="Headcount or headcount range.")
        location: str = SchemaField(description="Headquarters.")
        linkedin_url: str = SchemaField(description="Canonical page URL.")

    def __init__(self):
        super().__init__(
            id="a9b3e5d1-6c0f-4a28-b4e7-1d8c2f6a9e03",
            description=(
                "Read a LinkedIn company page through Treg, without a LinkedIn "
                "account or scraper."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "linkedin_url": "https://www.linkedin.com/company/modal-labs",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                *TEST_COMPANY_PAGE.items(),
                *routed_details(served_by="harvestapi.linkedin.company"),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    TEST_COMPANY_PAGE, served_by="harvestapi.linkedin.company"
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        routed = await self.route(
            credentials,
            "linkedin.company.profile",
            linkedin_reference(input_data.linkedin_url),
            input_data,
        )
        found = bool(routed.get("name"))
        if found:
            for output in typed_fields(routed.output, LINKEDIN_COMPANY_FIELDS):
                yield output
        for output in self.route_details(routed, found):
            yield output
