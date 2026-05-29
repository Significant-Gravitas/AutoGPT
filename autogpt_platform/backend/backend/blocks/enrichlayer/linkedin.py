"""
Block definitions for Enrichlayer API integration.

This module implements blocks for interacting with the Enrichlayer API,
which provides access to LinkedIn profile data and related information.
"""

import logging
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import APIKeyCredentials, CredentialsField, SchemaField
from backend.util.type import MediaFileType

from ._api import (
    EnrichlayerClient,
    Experience,
    FallbackToCache,
    PersonLookupResponse,
    PersonProfileResponse,
    RoleLookupResponse,
    UseCache,
)
from ._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, EnrichlayerCredentialsInput

logger = logging.getLogger(__name__)


class GetLinkedinProfileBlock(Block):
    """Block to fetch LinkedIn profile data using Enrichlayer API."""

    class Input(BlockSchemaInput):
        """Input schema for GetLinkedinProfileBlock."""

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
        credentials: EnrichlayerCredentialsInput = CredentialsField(
            description="Enrichlayer API credentials"
        )

    class Output(BlockSchemaOutput):
        """Output schema for GetLinkedinProfileBlock."""

        profile: PersonProfileResponse = SchemaField(
            description="LinkedIn profile data"
        )

    def __init__(self):
        """Initialize GetLinkedinProfileBlock."""
        super().__init__(
            id="f6e0ac73-4f1d-4acb-b4b7-b67066c5984e",
            description="Fetch LinkedIn profile data using Enrichlayer",
            categories={BlockCategory.SOCIAL},
            input_schema=GetLinkedinProfileBlock.Input,
            output_schema=GetLinkedinProfileBlock.Output,
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
            test_mock={
                "_fetch_profile": lambda *args, **kwargs: PersonProfileResponse(
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
            },
        )

    @staticmethod
    async def _fetch_profile(
        credentials: APIKeyCredentials,
        linkedin_url: str,
        fallback_to_cache: FallbackToCache = FallbackToCache.ON_ERROR,
        use_cache: UseCache = UseCache.IF_PRESENT,
        include_skills: bool = False,
        include_inferred_salary: bool = False,
        include_personal_email: bool = False,
        include_personal_contact_number: bool = False,
        include_social_media: bool = False,
        include_extra: bool = False,
    ):
        client = EnrichlayerClient(credentials)
        profile = await client.fetch_profile(
            linkedin_url=linkedin_url,
            fallback_to_cache=fallback_to_cache,
            use_cache=use_cache,
            include_skills=include_skills,
            include_inferred_salary=include_inferred_salary,
            include_personal_email=include_personal_email,
            include_personal_contact_number=include_personal_contact_number,
            include_social_media=include_social_media,
            include_extra=include_extra,
        )
        return profile

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the block to fetch LinkedIn profile data.

        Args:
            input_data: Input parameters for the block
            credentials: API key credentials for Enrichlayer
            **kwargs: Additional keyword arguments

        Yields:
            Tuples of (output_name, output_value)
        """
        try:
            profile = await self._fetch_profile(
                credentials=credentials,
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


class LinkedinPersonLookupBlock(Block):
    """Block to look up LinkedIn profiles by person's information using Enrichlayer API."""

    class Input(BlockSchemaInput):
        """Input schema for LinkedinPersonLookupBlock."""

        first_name: str = SchemaField(
            description="Person's first name",
            placeholder="John",
            advanced=False,
        )
        last_name: str | None = SchemaField(
            description="Person's last name",
            placeholder="Doe",
            default=None,
            advanced=False,
        )
        company_domain: str = SchemaField(
            description="Domain of the company they work for (optional)",
            placeholder="example.com",
            advanced=False,
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
        credentials: EnrichlayerCredentialsInput = CredentialsField(
            description="Enrichlayer API credentials"
        )

    class Output(BlockSchemaOutput):
        """Output schema for LinkedinPersonLookupBlock."""

        lookup_result: PersonLookupResponse = SchemaField(
            description="LinkedIn profile lookup result"
        )

    def __init__(self):
        """Initialize LinkedinPersonLookupBlock."""
        super().__init__(
            id="d237a98a-5c4b-4a1c-b9e3-e6f9a6c81df7",
            description="Look up LinkedIn profiles by person information using Enrichlayer",
            categories={BlockCategory.SOCIAL},
            input_schema=LinkedinPersonLookupBlock.Input,
            output_schema=LinkedinPersonLookupBlock.Output,
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
                        url="https://www.linkedin.com/in/williamhgates/",
                        name_similarity_score=0.93,
                        company_similarity_score=0.83,
                        title_similarity_score=0.3,
                        location_similarity_score=0.20,
                    ),
                )
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_lookup_person": lambda *args, **kwargs: PersonLookupResponse(
                    url="https://www.linkedin.com/in/williamhgates/",
                    name_similarity_score=0.93,
                    company_similarity_score=0.83,
                    title_similarity_score=0.3,
                    location_similarity_score=0.20,
                )
            },
        )

    @staticmethod
    async def _lookup_person(
        credentials: APIKeyCredentials,
        first_name: str,
        company_domain: str,
        last_name: str | None = None,
        location: Optional[str] = None,
        title: Optional[str] = None,
        include_similarity_checks: bool = False,
        enrich_profile: bool = False,
    ):
        client = EnrichlayerClient(credentials=credentials)
        lookup_result = await client.lookup_person(
            first_name=first_name,
            last_name=last_name,
            company_domain=company_domain,
            location=location,
            title=title,
            include_similarity_checks=include_similarity_checks,
            enrich_profile=enrich_profile,
        )
        return lookup_result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the block to look up LinkedIn profiles.

        Args:
            input_data: Input parameters for the block
            credentials: API key credentials for Enrichlayer
            **kwargs: Additional keyword arguments

        Yields:
            Tuples of (output_name, output_value)
        """
        try:
            lookup_result = await self._lookup_person(
                credentials=credentials,
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


class LinkedinRoleLookupBlock(Block):
    """Block to look up LinkedIn profiles by role in a company using Enrichlayer API."""

    class Input(BlockSchemaInput):
        """Input schema for LinkedinRoleLookupBlock."""

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
        credentials: EnrichlayerCredentialsInput = CredentialsField(
            description="Enrichlayer API credentials"
        )

    class Output(BlockSchemaOutput):
        """Output schema for LinkedinRoleLookupBlock."""

        role_lookup_result: RoleLookupResponse = SchemaField(
            description="LinkedIn role lookup result"
        )

    def __init__(self):
        """Initialize LinkedinRoleLookupBlock."""
        super().__init__(
            id="3b9fc742-06d4-49c7-b5ce-7e302dd7c8a7",
            description="Look up LinkedIn profiles by role in a company using Enrichlayer",
            categories={BlockCategory.SOCIAL},
            input_schema=LinkedinRoleLookupBlock.Input,
            output_schema=LinkedinRoleLookupBlock.Output,
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
            test_mock={
                "_lookup_role": lambda *args, **kwargs: RoleLookupResponse(
                    linkedin_profile_url="https://www.linkedin.com/in/williamhgates/",
                ),
            },
        )

    @staticmethod
    async def _lookup_role(
        credentials: APIKeyCredentials,
        role: str,
        company_name: str,
        enrich_profile: bool = False,
    ):
        client = EnrichlayerClient(credentials=credentials)
        role_lookup_result = await client.lookup_role(
            role=role,
            company_name=company_name,
            enrich_profile=enrich_profile,
        )
        return role_lookup_result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the block to look up LinkedIn profiles by role.

        Args:
            input_data: Input parameters for the block
            credentials: API key credentials for Enrichlayer
            **kwargs: Additional keyword arguments

        Yields:
            Tuples of (output_name, output_value)
        """
        try:
            role_lookup_result = await self._lookup_role(
                credentials=credentials,
                role=input_data.role,
                company_name=input_data.company_name,
                enrich_profile=input_data.enrich_profile,
            )
            yield "role_lookup_result", role_lookup_result
        except Exception as e:
            logger.error(f"Error looking up role in company: {str(e)}")
            yield "error", str(e)


class GetLinkedinProfilePictureBlock(Block):
    """Block to get LinkedIn profile pictures using Enrichlayer API."""

    class Input(BlockSchemaInput):
        """Input schema for GetLinkedinProfilePictureBlock."""

        linkedin_profile_url: str = SchemaField(
            description="LinkedIn profile URL",
            placeholder="https://www.linkedin.com/in/username/",
        )
        credentials: EnrichlayerCredentialsInput = CredentialsField(
            description="Enrichlayer API credentials"
        )

    class Output(BlockSchemaOutput):
        """Output schema for GetLinkedinProfilePictureBlock."""

        profile_picture_url: MediaFileType = SchemaField(
            description="LinkedIn profile picture URL"
        )

    def __init__(self):
        """Initialize GetLinkedinProfilePictureBlock."""
        super().__init__(
            id="68d5a942-9b3f-4e9a-b7c1-d96ea4321f0d",
            description="Get LinkedIn profile pictures using Enrichlayer",
            categories={BlockCategory.SOCIAL},
            input_schema=GetLinkedinProfilePictureBlock.Input,
            output_schema=GetLinkedinProfilePictureBlock.Output,
            test_input={
                "linkedin_profile_url": "https://www.linkedin.com/in/williamhgates/",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "profile_picture_url",
                    "https://media.licdn.com/dms/image/C4D03AQFj-xjuXrLFSQ/profile-displayphoto-shrink_800_800/0/1576881858598?e=1686787200&v=beta&t=zrQC76QwsfQQIWthfOnrKRBMZ5D-qIAvzLXLmWgYvTk",
                )
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_get_profile_picture": lambda *args, **kwargs: "https://media.licdn.com/dms/image/C4D03AQFj-xjuXrLFSQ/profile-displayphoto-shrink_800_800/0/1576881858598?e=1686787200&v=beta&t=zrQC76QwsfQQIWthfOnrKRBMZ5D-qIAvzLXLmWgYvTk",
            },
        )

    @staticmethod
    async def _get_profile_picture(
        credentials: APIKeyCredentials, linkedin_profile_url: str
    ):
        client = EnrichlayerClient(credentials=credentials)
        profile_picture_response = await client.get_profile_picture(
            linkedin_profile_url=linkedin_profile_url,
        )
        return profile_picture_response.profile_picture_url

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        """
        Run the block to get LinkedIn profile pictures.

        Args:
            input_data: Input parameters for the block
            credentials: API key credentials for Enrichlayer
            **kwargs: Additional keyword arguments

        Yields:
            Tuples of (output_name, output_value)
        """
        try:
            profile_picture = await self._get_profile_picture(
                credentials=credentials,
                linkedin_profile_url=input_data.linkedin_profile_url,
            )
            yield "profile_picture_url", profile_picture
        except Exception as e:
            logger.error(f"Error getting profile picture: {str(e)}")
            yield "error", str(e)
