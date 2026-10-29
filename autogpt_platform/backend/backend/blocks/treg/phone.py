"""Treg block that finds a person's phone number."""

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
from .emails import verify_header

PHONE_ADVICE = "This number was found, not confirmed live."


class TregFindPhoneNumberBlock(TregBlockBase):
    class Input(TregRoutedInput):
        linkedin_url: str = SchemaField(
            default="", description="The person's LinkedIn profile URL."
        )
        email: str = SchemaField(default="", description="Their email address.")
        full_name: str = SchemaField(
            default="", description="Their full name, with a company domain."
        )
        company_domain: str = SchemaField(
            default="", description="Their company's website domain."
        )
        verify: bool = SchemaField(
            default=False,
            description=(
                "Check that the number is live on its network in the same call. "
                "Only numbers with a country code can be checked, and a live line "
                "does not prove it is this person's."
            ),
        )

    class Output(TregRoutedOutput):
        phone: str = SchemaField(description="The phone number found.")
        line_type: str = SchemaField(description="mobile, landline, voip and so on.")
        country_code: str = SchemaField(description="The number's country code.")
        verification: str = SchemaField(
            description="The live check's verdict. Emitted only when it ran."
        )

    def __init__(self):
        super().__init__(
            id="e1f86c3b-0d7a-4b52-8c9e-3a5f1d7b2e68",
            description=(
                "Find a person's phone number from their LinkedIn URL, email, or name "
                "and company domain. Treg tries 16 providers cheapest first."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "linkedin_url": "https://www.linkedin.com/in/janedoe",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("phone", "+14155550123"),
                ("line_type", "mobile"),
                ("country_code", "US"),
                *routed_details(served_by="lusha.person.phone", advice=PHONE_ADVICE),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {
                        "phone": "+14155550123",
                        "line_type": "mobile",
                        "country_code": "US",
                    },
                    served_by="lusha.person.phone",
                    advice=PHONE_ADVICE,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        require_one(
            {
                "linkedin": bool(input_data.linkedin_url),
                "email": bool(input_data.email),
                "name": bool(input_data.full_name and input_data.company_domain),
            },
            "give a LinkedIn URL, an email, or a full name and company domain.",
        )
        routed = await self.route(
            credentials,
            "people.phone.find",
            {
                "linkedin_url": input_data.linkedin_url,
                "email": input_data.email,
                "full_name": input_data.full_name,
                "domain": input_data.company_domain,
            },
            input_data,
            headers=verify_header(input_data.verify),
        )
        found = bool(routed.get("phone"))
        if found:
            for output in typed_fields(
                routed.output, {"phone": str, "line_type": str, "country_code": str}
            ):
                yield output
            check = routed.meta.verification or {}
            if check.get("checked") and check.get("verdict"):
                yield "verification", check["verdict"]
        for output in self.route_details(routed, found):
            yield output
