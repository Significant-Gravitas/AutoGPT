"""Client, requests and error messages for the Google Search Console blocks."""

import json
from datetime import date
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from backend.util.exceptions import BlockExecutionError
from backend.util.settings import Settings

from ._auth import GoogleCredentials
from ._search_console_models import SearchConsoleFilter

WEBMASTERS_READONLY_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"

_QUOTA_REASONS = {
    "RATE_LIMIT_EXCEEDED",
    "dailyLimitExceeded",
    "quotaExceeded",
    "rateLimitExceeded",
    "userRateLimitExceeded",
}


def build_search_console_service(credentials: GoogleCredentials):
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
    return build("searchconsole", "v1", credentials=creds, cache_discovery=False)


def list_sites(service) -> list[dict[str, Any]]:
    """Every property the account can see, from sites.list."""
    return service.sites().list().execute().get("siteEntry", [])


def build_query_body(
    *,
    start_date: date,
    end_date: date,
    dimensions: list[str],
    search_type: str,
    filters: list[SearchConsoleFilter],
    row_limit: int,
    start_row: int,
    include_fresh_data: bool,
) -> dict[str, Any]:
    """The searchanalytics.query request body. All filters go in one AND group."""
    body: dict[str, Any] = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "dimensions": dimensions,
        "type": search_type,
        "dataState": "all" if include_fresh_data else "final",
        "rowLimit": row_limit,
        "startRow": start_row,
    }
    if filters:
        body["dimensionFilterGroups"] = [
            {"groupType": "and", "filters": [item.model_dump() for item in filters]}
        ]
    return body


def search_console_error(
    exc: HttpError,
    block_name: str,
    block_id: str,
    *,
    site_url: str = "",
    inspection_url: str = "",
) -> BlockExecutionError:
    """Turn a Search Console API error into a message the user can act on.

    ``site_url`` and ``inspection_url`` say what the request was about.
    """
    reason = str(exc.reason)
    lowered = reason.lower()
    status, reasons = _error_info(exc)
    if reasons & {"SERVICE_DISABLED", "accessNotConfigured"} or (
        exc.status_code == 403
        and ("has not been used" in lowered or "is disabled" in lowered)
    ):
        message = (
            "The Google Search Console API isn't enabled for this AutoGPT instance. "
            "Ask your AutoGPT administrator to enable it in the Google Cloud "
            f"project behind its Google sign-in. (Google said: {reason})"
        )
    elif "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in reasons or (
        exc.status_code == 403 and "insufficient" in lowered
    ):
        message = (
            "The connected Google account hasn't granted the Search Console access "
            "this block needs. Reconnect Google and approve Search Console access."
        )
    elif (
        exc.status_code == 429
        or status == "RESOURCE_EXHAUSTED"
        or reasons & _QUOTA_REASONS
        or "quota" in lowered
    ):
        message = _quota_message(reason, inspection_url)
    elif exc.status_code == 403:
        message = _access_message(reason, site_url, inspection_url)
    elif exc.status_code == 400:
        message = f"Google Search Console rejected the request: {reason}"
    else:
        message = f"Google Search Console API error {exc.status_code}: {reason}"
    return BlockExecutionError(
        message=message, block_name=block_name, block_id=block_id
    )


def _quota_message(reason: str, inspection_url: str) -> str:
    message = "Search Console's quota for this property is used up for now. "
    if inspection_url:
        message += (
            "URL inspection has its own limit of 2,000 inspections a day and 600 "
            "a minute per property. "
        )
    return f"{message}Try again later. (Google said: {reason})"


def _access_message(reason: str, site_url: str, inspection_url: str) -> str:
    target = (
        f"the Search Console property {site_url}"
        if site_url
        else "that Search Console property"
    )
    message = f"The connected Google account can't read {target}"
    if inspection_url:
        message += f", or {inspection_url} isn't part of it"
    return (
        f"{message}. Use Google Search Console List Sites to see the exact "
        "properties it can read (domain properties look like "
        f"sc-domain:example.com). (Google said: {reason})"
    )


def _error_info(exc: HttpError) -> tuple[str, set[str]]:
    """Google's status (such as PERMISSION_DENIED) and the reasons in the body."""
    try:
        error = json.loads(exc.content)["error"]
        entries = [*(error.get("details") or []), *(error.get("errors") or [])]
        reasons = {entry.get("reason") or "" for entry in entries}
        return str(error.get("status") or ""), reasons - {""}
    except (ValueError, KeyError, TypeError, AttributeError):
        return "", set()
