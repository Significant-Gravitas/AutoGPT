"""Shared helpers for the Google Analytics blocks: API clients, property IDs, errors.

The blocks read GA4 data with the Google Analytics Data API and list properties
with the Google Analytics Admin API, both under the analytics.readonly scope.
"""

import json
import re
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.settings import Settings

from ._auth import GoogleCredentials

ANALYTICS_READONLY_SCOPE = "https://www.googleapis.com/auth/analytics.readonly"
DATA_API = "Google Analytics Data API"
ADMIN_API = "Google Analytics Admin API"

PROPERTY_ID_DESCRIPTION = (
    "The Google Analytics 4 property's numeric ID, such as 123456789 (properties/"
    "123456789 works too). Find it in Google Analytics under Admin > Property "
    "details, or with Google Analytics List Properties. It isn't the G-... "
    "measurement ID."
)
FIELDS_HINT = "Check the names with Google Analytics List Dimensions and Metrics."

_FIND_PROPERTY_ID = (
    "Find it in Google Analytics under Admin > Property details, or use Google "
    "Analytics List Properties."
)
_PROPERTY_ID = re.compile(r"(?:properties/)?([0-9]+)")


def build_data_service(credentials: GoogleCredentials):
    return build(
        "analyticsdata",
        "v1beta",
        credentials=_google_credentials(credentials),
        cache_discovery=False,
    )


def build_admin_service(credentials: GoogleCredentials):
    return build(
        "analyticsadmin",
        "v1beta",
        credentials=_google_credentials(credentials),
        cache_discovery=False,
    )


def property_resource_name(value: str, block_name: str, block_id: str) -> str:
    """Accept 123456789 or properties/123456789; return properties/123456789."""
    value = value.strip()
    if match := _PROPERTY_ID.fullmatch(value):
        return f"properties/{match.group(1)}"
    given = value.removeprefix("properties/")
    if given.upper().startswith("G-"):
        message = (
            f"{given} is a measurement ID, the one used in website tags. The Google "
            "Analytics blocks need the property's numeric ID instead. "
            f"{_FIND_PROPERTY_ID}"
        )
    elif given.upper().startswith("UA-"):
        message = (
            f"{given} is a Universal Analytics ID. Universal Analytics was shut down "
            "in July 2024 and its data can no longer be read, so only Google "
            "Analytics 4 properties work. Use the GA4 property's numeric ID. "
            f"{_FIND_PROPERTY_ID}"
        )
    else:
        message = (
            "Enter a Google Analytics 4 property ID, a number such as 123456789. "
            f"{_FIND_PROPERTY_ID}"
        )
    raise BlockInputError(message=message, block_name=block_name, block_id=block_id)


def analytics_error(
    exc: HttpError,
    block_name: str,
    block_id: str,
    *,
    api: str,
    property_name: str = "",
    hint: str = FIELDS_HINT,
) -> BlockExecutionError:
    """Turn a Data or Admin API error into a message the user can act on.

    ``api`` names the API that was called, ``property_name`` the property the
    request was for, and ``hint`` is added to bad-request messages.
    """
    property_id = property_name.removeprefix("properties/")
    return BlockExecutionError(
        message=_error_message(exc, api, property_id, hint),
        block_name=block_name,
        block_id=block_id,
    )


def _error_message(exc: HttpError, api: str, property_id: str, hint: str) -> str:
    reason = str(exc.reason)
    error = _error_body(exc)
    reasons = {
        detail.get("reason")
        for detail in error.get("details") or []
        if isinstance(detail, dict)
    }
    status = exc.status_code
    if reasons & {"SERVICE_DISABLED", "ACCESS_NOT_CONFIGURED"} or (
        status == 403 and "has not been used" in reason.lower()
    ):
        return (
            f"The {api} isn't enabled for this AutoGPT instance. Ask your AutoGPT "
            "administrator to enable it in the Google Cloud project behind its "
            f"Google sign-in. (Google said: {reason})"
        )
    if (
        "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in reasons
        or "insufficient authentication scopes" in reason.lower()
    ):
        return (
            "The connected Google account hasn't granted the Google Analytics "
            "access this block needs. Reconnect Google and approve Google "
            "Analytics access."
        )
    if status == 429 or error.get("status") == "RESOURCE_EXHAUSTED":
        quota = f"quota for property {property_id}" if property_id else "API quota"
        return (
            f"The Google Analytics {quota} is used up for now. Try again later. "
            f"(Google said: {reason})"
        )
    if status == 403 and property_id:
        return (
            "The connected Google account can't read Google Analytics property "
            f"{property_id}. Check that it's a GA4 property ID, not an account or "
            "data stream ID, and that the account has at least Viewer access to "
            "the property. Google Analytics List Properties shows the properties "
            f"it can read. (Google said: {reason})"
        )
    if status == 400:
        return f"Google Analytics rejected the request: {reason} {hint}".rstrip()
    return f"Google Analytics API error {status}: {reason}"


def _error_body(exc: HttpError) -> dict[str, Any]:
    """The JSON error object Google sent, or {} when the body isn't one."""
    try:
        body = json.loads(exc.content)
    except ValueError:
        return {}
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def _google_credentials(credentials: GoogleCredentials) -> Credentials:
    settings = Settings()
    return Credentials(
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
