"""Shared models and helpers for the Google Contacts blocks (People API)."""

import time
from typing import Any, Callable, Iterable, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.settings import Settings

from ._auth import GoogleCredentials

CONTACTS_READONLY_SCOPE = "https://www.googleapis.com/auth/contacts.readonly"
OTHER_CONTACTS_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/contacts.other.readonly"
)
DIRECTORY_READONLY_SCOPE = "https://www.googleapis.com/auth/directory.readonly"
USERINFO_PROFILE_SCOPE = "https://www.googleapis.com/auth/userinfo.profile"
USERINFO_EMAIL_SCOPE = "https://www.googleapis.com/auth/userinfo.email"
ORGANIZATION_READ_SCOPE = "https://www.googleapis.com/auth/user.organization.read"

PERSON_FIELDS = "names,emailAddresses,phoneNumbers,organizations,photos"
# otherContacts.search rejects any field outside names, emailAddresses,
# phoneNumbers and metadata.
OTHER_CONTACT_FIELDS = "names,emailAddresses,phoneNumbers"
PROFILE_FIELDS = "names,emailAddresses,photos,organizations,locales"

SEARCH_RETRY_DELAY_SECONDS = 2.0

SearchFn = Callable[[str], list[dict[str, Any]]]


class GooglePerson(BaseModel):
    """A person from Google Contacts, the Workspace directory or a profile."""

    resource_name: str = Field(
        description=(
            "People API resource name: people/... for saved contacts, directory "
            "entries and profiles, otherContacts/... for other contacts"
        )
    )
    name: Optional[str] = Field(default=None, description="Display name")
    given_name: Optional[str] = Field(default=None, description="First name")
    family_name: Optional[str] = Field(default=None, description="Last name")
    email: Optional[str] = Field(default=None, description="Primary email address")
    emails: list[str] = Field(
        default_factory=list, description="All email addresses, primary first"
    )
    phones: list[str] = Field(
        default_factory=list,
        description=(
            "Phone numbers, primary first, in international format when Google "
            "can tell the country"
        ),
    )
    organization: Optional[str] = Field(
        default=None, description="Company or organization"
    )
    title: Optional[str] = Field(
        default=None, description="Job title at that organization"
    )
    photo_url: Optional[str] = Field(
        default=None,
        description="Photo URL (empty when the person only has Google's default avatar)",
    )


def build_people_service(credentials: GoogleCredentials):
    settings = Settings()
    creds = Credentials(
        token=(
            credentials.access_token.get_secret_value()
            if credentials.access_token
            else None
        ),
        refresh_token=(
            credentials.refresh_token.get_secret_value()
            if credentials.refresh_token
            else None
        ),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.secrets.google_client_id,
        client_secret=settings.secrets.google_client_secret,
        scopes=credentials.scopes,
    )
    return build("people", "v1", credentials=creds, cache_discovery=False)


def to_person(person: dict[str, Any]) -> GooglePerson:
    """Map a People API person resource to a GooglePerson."""
    name = primary_item(person.get("names"))
    organization = primary_item(person.get("organizations"))
    photo = primary_item(
        [photo for photo in person.get("photos") or [] if not photo.get("default")]
    )
    emails = _unique(
        item.get("value") for item in _primary_first(person.get("emailAddresses"))
    )
    phones = _unique(
        item.get("canonicalForm") or item.get("value")
        for item in _primary_first(person.get("phoneNumbers"))
    )
    return GooglePerson(
        resource_name=person.get("resourceName", ""),
        name=name.get("displayName"),
        given_name=name.get("givenName"),
        family_name=name.get("familyName"),
        email=emails[0] if emails else None,
        emails=emails,
        phones=phones,
        organization=organization.get("name"),
        title=organization.get("title"),
        photo_url=photo.get("url"),
    )


def primary_item(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Return the primary value of a person field, or its first, or {}."""
    ordered = _primary_first(items)
    return ordered[0] if ordered else {}


def search_with_warmup(
    searches: list[SearchFn], query: str
) -> list[list[dict[str, Any]]]:
    """Run contact searches the way the People API needs them run.

    Google answers contact searches from a cache it refreshes lazily, after a
    request, so a search can miss recent changes or find no one on a cold
    cache. Each search is first sent with an empty query to start a refresh.
    If no search finds anyone, they are all repeated once after a pause.
    """
    for search in searches:
        search("")
    found = [search(query) for search in searches]
    if any(found):
        return found
    time.sleep(SEARCH_RETRY_DELAY_SECONDS)
    return [search(query) for search in searches]


def require_query(query: str, block_name: str, block_id: str) -> str:
    query = query.strip()
    if not query:
        raise BlockInputError(
            message="Enter a name, email address or phone number to search for.",
            block_name=block_name,
            block_id=block_id,
        )
    return query


def people_error(
    exc: HttpError, block_name: str, block_id: str, *, access: str
) -> BlockExecutionError:
    """Turn a People API error into a message the user can act on.

    ``access`` names what the block reads, e.g. "its contacts".
    """
    reason = str(exc.reason)
    if exc.status_code == 403 and "insufficient" in reason.lower():
        message = (
            f"The connected Google account hasn't granted access to {access}. "
            "Reconnect Google and approve it."
        )
    elif exc.status_code == 400 and "domain user" in reason.lower():
        message = (
            "Directory search only works for Google Workspace (work or school) "
            "accounts, and the connected account is a personal Google account. "
            "Use Google Contacts Search to search its contacts instead."
        )
    else:
        message = f"Google People API error {exc.status_code}: {reason}"
    return BlockExecutionError(
        message=message, block_name=block_name, block_id=block_id
    )


def _primary_first(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return sorted(
        items or [], key=lambda item: not item.get("metadata", {}).get("primary")
    )


def _unique(values: Iterable[str | None]) -> list[str]:
    """Drop empty and repeated values, comparing without case."""
    by_key: dict[str, str] = {}
    for value in values:
        if value:
            by_key.setdefault(value.lower(), value)
    return list(by_key.values())
