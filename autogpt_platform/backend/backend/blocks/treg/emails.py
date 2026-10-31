"""Treg blocks that find and check a person's email address."""

from collections.abc import Iterator
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
    TregRouted,
    TregRoutedInput,
    TregRoutedOutput,
    require_one,
    typed_fields,
)
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import mock_routed, routed_details

VERIFY_DESCRIPTION = (
    "Check that the address accepts mail in the same call (a fraction of a cent "
    "more). A found address is a provider's best match until it is checked."
)


class _VerifiedEmailOutput(TregRoutedOutput):
    email: str = SchemaField(description="The email address found.")
    verified: bool = SchemaField(
        description="True only when the mailbox was checked and accepts mail."
    )
    verification: str = SchemaField(
        description="The check's verdict: valid, invalid, catch_all, risky or "
        "unknown. Emitted only when the address was checked."
    )


def email_verification(routed: TregRouted) -> tuple[bool, str | None]:
    """Whether a found address is confirmed, and the check's verdict if run."""
    check = routed.meta.verification or {}
    verdict = check.get("verdict") if check.get("checked") else None
    if verdict:
        return verdict == "valid", verdict
    return routed.get("verified") is True, None


def verify_header(verify: bool) -> dict[str, str]:
    return {"X-Treg-Route-Verify": "true"} if verify else {}


def found_email(routed: TregRouted) -> str | None:
    """The address a finder returned, if it is a non-empty string."""
    email = routed.get("email")
    return email if isinstance(email, str) and email.strip() else None


def email_outputs(
    routed: TregRouted, extra: dict[str, type]
) -> Iterator[tuple[str, Any]]:
    """The found address, whether it is confirmed, and any extra fields."""
    email = found_email(routed)
    if email is None:
        return
    verified, verdict = email_verification(routed)
    yield "email", email
    yield "verified", verified
    if verdict:
        yield "verification", verdict
    yield from typed_fields(routed.output, extra)


class TregFindWorkEmailBlock(TregBlockBase):
    class Input(TregRoutedInput):
        full_name: str = SchemaField(
            default="", description="The person's full name, e.g. 'Jane Doe'."
        )
        company_domain: str = SchemaField(
            default="",
            description="Their company's website domain, e.g. 'outerbounds.com'.",
        )
        linkedin_url: str = SchemaField(
            default="",
            description="Their LinkedIn profile URL. Enough on its own, without "
            "a name and domain.",
        )
        verify: bool = SchemaField(default=True, description=VERIFY_DESCRIPTION)

    class Output(_VerifiedEmailOutput):
        confidence: float = SchemaField(
            description="The provider's confidence in the match, from 0 to 1."
        )
        first_name: str = SchemaField(description="First name, as the provider has it.")
        last_name: str = SchemaField(description="Last name, as the provider has it.")

    def __init__(self):
        super().__init__(
            id="b7e2d0a4-3f51-4c86-9e1a-6d4b8f2c7a90",
            description=(
                "Find a person's work email from their name and company domain, or "
                "their LinkedIn URL. Treg tries providers cheapest first and can "
                "check the address before you send to it."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "full_name": "Jane Doe",
                "company_domain": "outerbounds.com",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("email", "jane@outerbounds.com"),
                ("verified", True),
                ("verification", "valid"),
                ("confidence", 0.94),
                ("first_name", "Jane"),
                ("last_name", "Doe"),
                *routed_details(),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {
                        "email": "jane@outerbounds.com",
                        "confidence": 0.94,
                        "first_name": "Jane",
                        "last_name": "Doe",
                        "verified": False,
                    },
                    verification={"verdict": "valid", "checked": True},
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
                "name": bool(input_data.full_name and input_data.company_domain),
            },
            "give a full name and company domain, or a LinkedIn URL.",
        )
        routed = await self.route(
            credentials,
            "people.email.find",
            {
                "full_name": input_data.full_name,
                "domain": input_data.company_domain,
                "linkedin_url": input_data.linkedin_url,
            },
            input_data,
            headers=verify_header(input_data.verify),
        )
        for output in email_outputs(routed, {"confidence": float}):
            yield output
        if found_email(routed) is not None:
            for output in typed_fields(
                routed.output, {"first_name": str, "last_name": str}
            ):
                yield output
        for output in self.route_details(routed, found_email(routed) is not None):
            yield output


class TregFindPersonalEmailBlock(TregBlockBase):
    class Input(TregRoutedInput):
        linkedin_url: str = SchemaField(
            default="", description="The person's LinkedIn profile URL."
        )
        work_email: str = SchemaField(
            default="", description="Their work email, if you have it instead."
        )
        verify: bool = SchemaField(default=True, description=VERIFY_DESCRIPTION)

    class Output(_VerifiedEmailOutput):
        pass

    def __init__(self):
        super().__init__(
            id="4c9a1e7f-8b2d-4e63-a5f0-2d7c9b1e8a34",
            description=(
                "Find a person's personal email (Gmail, Outlook and so on) from their "
                "LinkedIn URL or work email, through Treg."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "linkedin_url": "https://www.linkedin.com/in/janedoe",
                "verify": False,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("email", "jane.doe@gmail.com"),
                ("verified", False),
                *routed_details(served_by="leadmagic.x.personal-email"),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {"email": "jane.doe@gmail.com", "verified": None},
                    served_by="leadmagic.x.personal-email",
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
                "email": bool(input_data.work_email),
            },
            "give a LinkedIn URL or a work email.",
        )
        routed = await self.route(
            credentials,
            "people.email.find.personal",
            {"linkedin_url": input_data.linkedin_url, "email": input_data.work_email},
            input_data,
            headers=verify_header(input_data.verify),
        )
        for output in email_outputs(routed, {}):
            yield output
        for output in self.route_details(routed, found_email(routed) is not None):
            yield output


class TregVerifyEmailBlock(TregBlockBase):
    class Input(TregRoutedInput):
        email: str = SchemaField(
            description="The email address to check.", min_length=3
        )

    class Output(TregRoutedOutput):
        valid: bool = SchemaField(description="True when the mailbox accepts mail.")
        status: str = SchemaField(
            description="The provider's verdict, e.g. valid, invalid, accept_all "
            "(the domain accepts any address, so this one is unconfirmed) or unknown."
        )
        score: int = SchemaField(description="The provider's deliverability score.")

    def __init__(self):
        super().__init__(
            id="9d3b7f12-6a4e-4c08-b1d5-7e2a9c4f6b13",
            description=(
                "Check whether an email address accepts mail before you send to it, "
                "through Treg's cheapest available verifier."
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
                ("valid", True),
                ("status", "valid"),
                ("score", 97),
                *routed_details(served_by="bounceban.verify", cost_usd=0.0008),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_routed(
                    {"valid": True, "status": "valid", "score": 97},
                    served_by="bounceban.verify",
                    cost_usd=0.0008,
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        routed = await self.route(
            credentials, "people.email.verify", {"email": input_data.email}, input_data
        )
        # Treg's contract: a verification miss is a null status.
        found = routed.get("status") is not None
        if found:
            fields: dict[str, Any] = {"valid": bool, "status": str, "score": int}
            for output in typed_fields(routed.output, fields):
                yield output
        for output in self.route_details(routed, found):
            yield output
