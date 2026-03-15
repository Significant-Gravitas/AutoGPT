"""
API module for Enrichlayer integration.

This module provides a client for interacting with the Enrichlayer API,
which allows fetching LinkedIn profile data and related information.
"""

import datetime
import enum
import logging
from json import JSONDecodeError
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, Field

from backend.data.model import APIKeyCredentials
from backend.util.request import Requests

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EnrichlayerAPIException(Exception):
    """Exception raised for Enrichlayer API errors."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class FallbackToCache(enum.Enum):
    ON_ERROR = "on-error"
    NEVER = "never"


class UseCache(enum.Enum):
    IF_PRESENT = "if-present"
    NEVER = "never"


class SocialMediaProfiles(BaseModel):
    """Social media profiles model."""

    twitter: Optional[str] = None
    facebook: Optional[str] = None
    github: Optional[str] = None


class Experience(BaseModel):
    """Experience model for LinkedIn profiles."""

    company: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    starts_at: Optional[dict[str, int]] = None
    ends_at: Optional[dict[str, int]] = None
    company_linkedin_profile_url: Optional[str] = None


class Education(BaseModel):
    """Education model for LinkedIn profiles."""

    school: Optional[str] = None
    degree_name: Optional[str] = None
    field_of_study: Optional[str] = None
    starts_at: Optional[dict[str, int]] = None
    ends_at: Optional[dict[str, int]] = None
    school_linkedin_profile_url: Optional[str] = None


class PersonProfileResponse(BaseModel):
    """Response model for LinkedIn person profile.

    This model represents the response from Enrichlayer's LinkedIn profile API.
    The API returns comprehensive profile data including work experience,
    education, skills, and contact information (when available).

    Example API Response:
    {
        "public_identifier": "johnsmith",
        "full_name": "John Smith",
        "occupation": "Software Engineer at Tech Corp",
        "experiences": [
            {
                "company": "Tech Corp",
                "title": "Software Engineer",
                "starts_at": {"year": 2020, "month": 1}
            }
        ],
        "education": [...],
        "skills": ["Python", "JavaScript", ...]
    }
    """

    public_identifier: Optional[str] = None
    profile_pic_url: Optional[str] = None
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    occupation: Optional[str] = None
    headline: Optional[str] = None
    summary: Optional[str] = None
    country: Optional[str] = None
    country_full_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    experiences: Optional[list[Experience]] = None
    education: Optional[list[Education]] = None
    languages: Optional[list[str]] = None
    skills: Optional[list[str]] = None
    inferred_salary: Optional[dict[str, Any]] = None
    personal_email: Optional[str] = None
    personal_contact_number: Optional[str] = None
    social_media_profiles: Optional[SocialMediaProfiles] = None
    extra: Optional[dict[str, Any]] = None


class SimilarProfile(BaseModel):
    """Similar profile model for LinkedIn person lookup."""

    similarity: float
    linkedin_profile_url: str


class PersonLookupResponse(BaseModel):
    """Response model for LinkedIn person lookup.

    This model represents the response from Enrichlayer's person lookup API.
    The API returns a LinkedIn profile URL and similarity scores when
    searching for a person by name and company.

    Example API Response:
    {
        "url": "https://www.linkedin.com/in/johnsmith/",
        "name_similarity_score": 0.95,
        "company_similarity_score": 0.88,
        "title_similarity_score": 0.75,
        "location_similarity_score": 0.60
    }
    """

    url: str | None = None
    name_similarity_score: float | None
    company_similarity_score: float | None
    title_similarity_score: float | None
    location_similarity_score: float | None
    last_updated: datetime.datetime | None = None
    profile: PersonProfileResponse | None = None


class RoleLookupResponse(BaseModel):
    """Response model for LinkedIn role lookup.

    This model represents the response from Enrichlayer's role lookup API.
    The API returns LinkedIn profile data for a specific role at a company.

    Example API Response:
    {
        "linkedin_profile_url": "https://www.linkedin.com/in/johnsmith/",
        "profile_data": {...}  // Full PersonProfileResponse data when enrich_profile=True
    }
    """

    linkedin_profile_url: Optional[str] = None
    profile_data: Optional[PersonProfileResponse] = None


class ProfilePictureResponse(BaseModel):
    """Response model for LinkedIn profile picture.

    This model represents the response from Enrichlayer's profile picture API.
    The API returns a URL to the person's LinkedIn profile picture.

    Example API Response:
    {
        "tmp_profile_pic_url": "https://media.licdn.com/dms/image/..."
    }
    """

    tmp_profile_pic_url: str = Field(
        ..., description="URL of the profile picture", alias="tmp_profile_pic_url"
    )

    @property
    def profile_picture_url(self) -> str:
        """Backward compatibility property for profile_picture_url."""
        return self.tmp_profile_pic_url


class EnrichlayerClient:
    """Client for interacting with the Enrichlayer API."""

    API_BASE_URL = "https://enrichlayer.com/api/v2"

    def __init__(
        self,
        credentials: Optional[APIKeyCredentials] = None,
        custom_requests: Optional[Requests] = None,
    ):
        """
        Initialize the Enrichlayer client.

        Args:
            credentials: The credentials to use for authentication.
            custom_requests: Custom Requests instance for testing.
        """
        if custom_requests:
            self._requests = custom_requests
        else:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
            }
            if credentials:
                headers["Authorization"] = (
                    f"Bearer {credentials.api_key.get_secret_value()}"
                )

            self._requests = Requests(
                extra_headers=headers,
                raise_for_status=False,
            )

    async def _handle_response(self, response) -> Any:
        """
        Handle API response and check for errors.

        Args:
            response: The response object from the request.

        Returns:
            The response data.

        Raises:
            EnrichlayerAPIException: If the API request fails.
        """
        if not response.ok:
            try:
                error_data = response.json()
                error_message = error_data.get("message", "")
            except JSONDecodeError:
                error_message = response.text

            raise EnrichlayerAPIException(
                f"Enrichlayer API request failed ({response.status_code}): {error_message}",
                response.status_code,
            )

        return response.json()

    async def fetch_profile(
        self,
        linkedin_url: str,
        fallback_to_cache: FallbackToCache = FallbackToCache.ON_ERROR,
        use_cache: UseCache = UseCache.IF_PRESENT,
        include_skills: bool = False,
        include_inferred_salary: bool = False,
        include_personal_email: bool = False,
        include_personal_contact_number: bool = False,
        include_social_media: bool = False,
        include_extra: bool = False,
    ) -> PersonProfileResponse:
        """
        Fetch a LinkedIn profile with optional parameters.

        Args:
            linkedin_url: The LinkedIn profile URL to fetch.
            fallback_to_cache: Cache usage if live fetch fails ('on-error' or 'never').
            use_cache: Cache utilization ('if-present' or 'never').
            include_skills: Whether to include skills data.
            include_inferred_salary: Whether to include inferred salary data.
            include_personal_email: Whether to include personal email.
            include_personal_contact_number: Whether to include personal contact number.
            include_social_media: Whether to include social media profiles.
            include_extra: Whether to include additional data.

        Returns:
            The LinkedIn profile data.

        Raises:
            EnrichlayerAPIException: If the API request fails.
        """
        params = {
            "url": linkedin_url,
            "fallback_to_cache": fallback_to_cache.value.lower(),
            "use_cache": use_cache.value.lower(),
        }

        if include_skills:
            params["skills"] = "include"
        if include_inferred_salary:
            params["inferred_salary"] = "include"
        if include_personal_email:
            params["personal_email"] = "include"
        if include_personal_contact_number:
            params["personal_contact_number"] = "include"
        if include_social_media:
            params["twitter_profile_id"] = "include"
            params["facebook_profile_id"] = "include"
            params["github_profile_id"] = "include"
        if include_extra:
            params["extra"] = "include"

        response = await self._requests.get(
            f"{self.API_BASE_URL}/profile", params=params
        )
        return PersonProfileResponse(**await self._handle_response(response))

    async def lookup_person(
        self,
        first_name: str,
        company_domain: str,
        last_name: str | None = None,
        location: Optional[str] = None,
        title: Optional[str] = None,
        include_similarity_checks: bool = False,
        enrich_profile: bool = False,
    ) -> PersonLookupResponse:
        """
        Look up a LinkedIn profile by person's information.

        Args:
            first_name: The person's first name.
            last_name: The person's last name.
            company_domain: The domain of the company they work for.
            location: The person's location.
            title: The person's job title.
            include_similarity_checks: Whether to include similarity checks.
            enrich_profile: Whether to enrich the profile.

        Returns:
            The LinkedIn profile lookup result.

        Raises:
            EnrichlayerAPIException: If the API request fails.
        """
        params = {"first_name": first_name, "company_domain": company_domain}

        if last_name:
            params["last_name"] = last_name
        if location:
            params["location"] = location
        if title:
            params["title"] = title
        if include_similarity_checks:
            params["similarity_checks"] = "include"
        if enrich_profile:
            params["enrich_profile"] = "enrich"

        response = await self._requests.get(
            f"{self.API_BASE_URL}/profile/resolve", params=params
        )
        return PersonLookupResponse(**await self._handle_response(response))

    async def lookup_role(
        self, role: str, company_name: str, enrich_profile: bool = False
    ) -> RoleLookupResponse:
        """
        Look up a LinkedIn profile by role in a company.

        Args:
            role: The role title (e.g., CEO, CTO).
            company_name: The name of the company.
            enrich_profile: Whether to enrich the profile.

        Returns:
            The LinkedIn profile lookup result.

        Raises:
            EnrichlayerAPIException: If the API request fails.
        """
        params = {
            "role": role,
            "company_name": company_name,
        }

        if enrich_profile:
            params["enrich_profile"] = "enrich"

        response = await self._requests.get(
            f"{self.API_BASE_URL}/find/company/role", params=params
        )
        return RoleLookupResponse(**await self._handle_response(response))

    async def get_profile_picture(
        self, linkedin_profile_url: str
    ) -> ProfilePictureResponse:
        """
        Get a LinkedIn profile picture URL.

        Args:
            linkedin_profile_url: The LinkedIn profile URL.

        Returns:
            The profile picture URL.

        Raises:
            EnrichlayerAPIException: If the API request fails.
        """
        params = {
            "linkedin_person_profile_url": linkedin_profile_url,
        }

        response = await self._requests.get(
            f"{self.API_BASE_URL}/person/profile-picture", params=params
        )
        return ProfilePictureResponse(**await self._handle_response(response))
