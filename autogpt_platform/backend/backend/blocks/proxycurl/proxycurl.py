"""
Block definitions for Proxycurl API integration.

This module implements blocks for interacting with the Proxycurl API,
which provides access to LinkedIn profile data and related information.
"""

import logging
from typing import Optional

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import APIKeyCredentials, CredentialsField, SchemaField

from ._api import (
    Experience,
    FallbackToCache,
    PersonLookupResponse,
    PersonProfileResponse,
    ProfilePictureResponse,
    ProxycurlClient,
    RoleLookupResponse,
    SimilarProfile,
    UseCache,
)
from ._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, ProxycurlCredentialsInput

logger = logging.getLogger(__name__)


class ProxycurlProfileFetchBlock(Block):
    """Block to fetch LinkedIn profile data using Proxycurl API."""

    class Input(BlockSchema):
        """Input schema for ProxycurlProfileFetchBlock."""

        linkedin_url: str = SchemaField(
            description="LinkedIn profile URL to fetch data from",
            placeholder="https://www.linkedin.com/in/username/",
        )
        fallback_to_cache: FallbackToCache = SchemaField(
            description="Cache usage if live fetch fails",
            default=FallbackToCache.ON_ERROR,
            advanced=True,
        )
        use_cache: UseCache = SchemaField(
            description="Cache utilization strategy",
            default=UseCache.IF_PRESENT,
            advanced=True,
        )
        include_skills: bool = SchemaField(
            description="Include skills data",
            default=False,
            advanced=True,
        )
        include_inferred_salary: bool = SchemaField(
            description="Include inferred salary data",
            default=False,
            advanced=True,
        )
        include_personal_email: bool = SchemaField(
            description="Include personal email",
            default=False,
            advanced=True,
        )
        include_personal_contact_number: bool = SchemaField(
            description="Include personal contact number",
            default=False,
            advanced=True,
        )
        include_social_media: bool = SchemaField(
            description="Include social media profiles",
            default=False,
            advanced=True,
        )
        include_extra: bool = SchemaField(
            description="Include additional data",
            default=False,
            advanced=True,
        )
        credentials: ProxycurlCredentialsInput = CredentialsField(
            description="Proxycurl API credentials"
        )

    class Output(BlockSchema):
        """Output schema for ProxycurlProfileFetchBlock."""

        profile: PersonProfileResponse = SchemaField(
            description="LinkedIn profile data"
        )
        error: Optional[str] = SchemaField(
            description="Error message if the request failed"
        )

    def __init__(self):
        """Initialize ProxycurlProfileFetchBlock."""
        super().__init__(
            id="f6e0ac73-4f1d-4acb-b4b7-b67066c5984e",
            description="Fetch LinkedIn profile data using Proxycurl",
            categories={BlockCategory.SOCIAL},
            input_schema=ProxycurlProfileFetchBlock.Input,
            output_schema=ProxycurlProfileFetchBlock.Output,
            test_input={
                "linkedin_url": "https://www.linkedin.com/in/williamhgates/",
                "include_skills": True,
                "include_social_media": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "profile",
                    PersonProfileResponse(
                        public_identifier="williamhgates",
                        full_name="Bill Gates",
                        occupation="Co-chair at Bill & Melinda Gates Foundation",
                        experiences=[
                            Experience(
                                company="Bill & Melinda Gates Foundation",
                                title="Co-chair",
                                starts_at={"year": 2000},
                            )
                        ],
                    ),
                )
            ],
            test_credentials=TEST_CREDENTIALS,
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the block to fetch LinkedIn profile data.

        Args:
            input_data: Input parameters for the block
            credentials: API key credentials for Proxycurl
            **kwargs: Additional keyword arguments

        Yields:
            Tuples of (output_name, output_value)
        """
        try:
            client = ProxycurlClient(credentials=credentials)
            profile = client.fetch_profile(
                linkedin_url=input_data.linkedin_url,
                fallback_to_cache=input_data.fallback_to_cache,
                use_cache=input_data.use_cache,
                include_skills=input_data.include_skills,
                include_inferred_salary=input_data.include_inferred_salary,
                include_personal_email=input_data.include_personal_email,
                include_personal_contact_number=input_data.include_personal_contact_number,
                include_social_media=input_data.include_social_media,
                include_extra=input_data.include_extra,
            )
            yield "profile", profile
        except Exception as e:
            logger.error(f"Error fetching LinkedIn profile: {str(e)}")
            yield "error", str(e)


class ProxycurlPersonLookupBlock(Block):
    """Block to look up LinkedIn profiles by person's information using Proxycurl API."""

    class Input(BlockSchema):
        """Input schema for ProxycurlPersonLookupBlock."""

        first_name: str = SchemaField(
            description="Person's first name",
            placeholder="John",
        )
        last_name: str = SchemaField(
            description="Person's last name",
            placeholder="Doe",
        )
        company_domain: Optional[str] = SchemaField(
            description="Domain of the company they work for (optional)",
            placeholder="example.com",
            default=None,
        )
        location: Optional[str] = SchemaField(
            description="Person's location (optional)",
            placeholder="San Francisco",
            default=None,
        )
        title: Optional[str] = SchemaField(
            description="Person's job title (optional)",
            placeholder="CEO",
            default=None,
        )
        include_similarity_checks: bool = SchemaField(
            description="Include similarity checks",
            default=False,
            advanced=True,
        )
        enrich_profile: bool = SchemaField(
            description="Enrich the profile with additional data",
            default=False,
            advanced=True,
        )
        credentials: ProxycurlCredentialsInput = CredentialsField(
            description="Proxycurl API credentials"
        )

    class Output(BlockSchema):
        """Output schema for ProxycurlPersonLookupBlock."""

        lookup_result: PersonLookupResponse = SchemaField(
            description="LinkedIn profile lookup result"
        )
        error: Optional[str] = SchemaField(
            description="Error message if the request failed"
        )

    def __init__(self):
        """Initialize ProxycurlPersonLookupBlock."""
        super().__init__(
            id="d237a98a-5c4b-4a1c-b9e3-e6f9a6c81df7",
            description="Look up LinkedIn profiles by person information using Proxycurl",
            categories={BlockCategory.SOCIAL},
            input_schema=ProxycurlPersonLookupBlock.Input,
            output_schema=ProxycurlPersonLookupBlock.Output,
            test_input={
                "first_name": "Bill",
                "last_name": "Gates",
                "company_domain": "gatesfoundation.org",
                "include_similarity_checks": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "lookup_result",
                    PersonLookupResponse(
                        linkedin_profile_url="https://www.linkedin.com/in/williamhgates/",
                        similar_profiles=[
                            SimilarProfile(
                                similarity=0.95,
                                linkedin_profile_url="https://www.linkedin.com/in/billgates/",
                            )
                        ],
                    ),
                )
            ],
            test_credentials=TEST_CREDENTIALS,
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the block to look up LinkedIn profiles.

        Args:
            input_data: Input parameters for the block
            credentials: API key credentials for Proxycurl
            **kwargs: Additional keyword arguments

        Yields:
            Tuples of (output_name, output_value)
        """
        try:
            client = ProxycurlClient(credentials=credentials)
            lookup_result = client.lookup_person(
                first_name=input_data.first_name,
                last_name=input_data.last_name,
                company_domain=input_data.company_domain,
                location=input_data.location,
                title=input_data.title,
                include_similarity_checks=input_data.include_similarity_checks,
                enrich_profile=input_data.enrich_profile,
            )
            yield "lookup_result", lookup_result
        except Exception as e:
            logger.error(f"Error looking up LinkedIn profile: {str(e)}")
            yield "error", str(e)


class ProxycurlRoleLookupBlock(Block):
    """Block to look up LinkedIn profiles by role in a company using Proxycurl API."""

    class Input(BlockSchema):
        """Input schema for ProxycurlRoleLookupBlock."""

        role: str = SchemaField(
            description="Role title (e.g., CEO, CTO)",
            placeholder="CEO",
        )
        company_name: str = SchemaField(
            description="Name of the company",
            placeholder="Microsoft",
        )
        enrich_profile: bool = SchemaField(
            description="Enrich the profile with additional data",
            default=False,
            advanced=True,
        )
        credentials: ProxycurlCredentialsInput = CredentialsField(
            description="Proxycurl API credentials"
        )

    class Output(BlockSchema):
        """Output schema for ProxycurlRoleLookupBlock."""

        role_lookup_result: RoleLookupResponse = SchemaField(
            description="LinkedIn role lookup result"
        )
        error: Optional[str] = SchemaField(
            description="Error message if the request failed"
        )

    def __init__(self):
        """Initialize ProxycurlRoleLookupBlock."""
        super().__init__(
            id="3b9fc742-06d4-49c7-b5ce-7e302dd7c8a7",
            description="Look up LinkedIn profiles by role in a company using Proxycurl",
            categories={BlockCategory.SOCIAL},
            input_schema=ProxycurlRoleLookupBlock.Input,
            output_schema=ProxycurlRoleLookupBlock.Output,
            test_input={
                "role": "Co-chair",
                "company_name": "Gates Foundation",
                "enrich_profile": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "role_lookup_result",
                    RoleLookupResponse(
                        linkedin_profile_url="https://www.linkedin.com/in/williamhgates/",
                    ),
                )
            ],
            test_credentials=TEST_CREDENTIALS,
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the block to look up LinkedIn profiles by role.

        Args:
            input_data: Input parameters for the block
            credentials: API key credentials for Proxycurl
            **kwargs: Additional keyword arguments

        Yields:
            Tuples of (output_name, output_value)
        """
        try:
            client = ProxycurlClient(credentials=credentials)
            role_lookup_result = client.lookup_role(
                role=input_data.role,
                company_name=input_data.company_name,
                enrich_profile=input_data.enrich_profile,
            )
            yield "role_lookup_result", role_lookup_result
        except Exception as e:
            logger.error(f"Error looking up role in company: {str(e)}")
            yield "error", str(e)


class ProxycurlProfilePictureBlock(Block):
    """Block to get LinkedIn profile pictures using Proxycurl API."""

    class Input(BlockSchema):
        """Input schema for ProxycurlProfilePictureBlock."""

        linkedin_profile_url: str = SchemaField(
            description="LinkedIn profile URL",
            placeholder="https://www.linkedin.com/in/username/",
        )
        credentials: ProxycurlCredentialsInput = CredentialsField(
            description="Proxycurl API credentials"
        )

    class Output(BlockSchema):
        """Output schema for ProxycurlProfilePictureBlock."""

        profile_picture: ProfilePictureResponse = SchemaField(
            description="LinkedIn profile picture URL"
        )
        error: Optional[str] = SchemaField(
            description="Error message if the request failed"
        )

    def __init__(self):
        """Initialize ProxycurlProfilePictureBlock."""
        super().__init__(
            id="68d5a942-9b3f-4e9a-b7c1-d96ea4321f0d",
            description="Get LinkedIn profile pictures using Proxycurl",
            categories={BlockCategory.SOCIAL},
            input_schema=ProxycurlProfilePictureBlock.Input,
            output_schema=ProxycurlProfilePictureBlock.Output,
            test_input={
                "linkedin_profile_url": "https://www.linkedin.com/in/williamhgates/",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "profile_picture",
                    ProfilePictureResponse(
                        profile_picture_url="https://media.licdn.com/dms/image/C4D03AQFj-xjuXrLFSQ/profile-displayphoto-shrink_800_800/0/1576881858598?e=1686787200&v=beta&t=zrQC76QwsfQQIWthfOnrKRBMZ5D-qIAvzLXLmWgYvTk"
                    ),
                )
            ],
            test_credentials=TEST_CREDENTIALS,
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the block to get LinkedIn profile pictures.

        Args:
            input_data: Input parameters for the block
            credentials: API key credentials for Proxycurl
            **kwargs: Additional keyword arguments

        Yields:
            Tuples of (output_name, output_value)
        """
        try:
            client = ProxycurlClient(credentials=credentials)
            profile_picture = client.get_profile_picture(
                linkedin_profile_url=input_data.linkedin_profile_url,
            )
            yield "profile_picture", profile_picture
        except Exception as e:
            logger.error(f"Error getting profile picture: {str(e)}")
            yield "error", str(e)
