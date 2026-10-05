"""Turn the Search Console blocks' site and date inputs into what the API takes."""

import asyncio
import re
from datetime import date, datetime, timedelta
from typing import Any, Callable
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from googleapiclient.errors import HttpError

from backend.util.exceptions import BlockExecutionError, BlockInputError

from ._search_console_api import search_console_error
from ._search_console_models import SearchConsoleSite, to_site

SITE_URL_DESCRIPTION = (
    "The Search Console property, as Search Console lists it: "
    "sc-domain:example.com for a domain property or https://www.example.com/ "
    "for a URL-prefix property. A bare domain like example.com also works: the "
    "block picks the matching property the account can read."
)

# Search Console's days run on Pacific Time, so relative dates do too.
SEARCH_CONSOLE_TIME_ZONE = ZoneInfo("America/Los_Angeles")
UNVERIFIED_LEVEL = "siteUnverifiedUser"
MAX_LISTED_SITES = 10

_DAYS_AGO = re.compile(r"(\d+)daysago")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DOMAIN = re.compile(r"[^\s/:@]+(?:\.[^\s/:@]+)+")


async def resolve_site_url(
    value: str,
    list_sites: Callable[[], list[dict[str, Any]]],
    block_name: str,
    block_id: str,
    *,
    inspection_url: str = "",
) -> str:
    """The property to call the API with.

    A property is used as given. For a bare domain, ``list_sites`` is called
    once and the domain is matched against the properties the account can read.
    With ``inspection_url``, only properties that contain that URL match.
    """
    if site_url := as_property(value):
        return site_url
    domain = _bare_domain(value, block_name, block_id)
    try:
        entries = await asyncio.to_thread(list_sites)
    except HttpError as e:
        raise search_console_error(e, block_name, block_id) from e
    sites = [to_site(entry) for entry in entries]
    if site_url := match_site(domain, sites, inspection_url):
        return site_url
    raise BlockExecutionError(
        message=_no_match_message(domain, sites, inspection_url),
        block_name=block_name,
        block_id=block_id,
    )


def require_page_url(value: str, block_name: str, block_id: str) -> str:
    url = value.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise BlockInputError(
            message=(
                "Give the full URL of the page to inspect, starting with https:// "
                "or http://."
            ),
            block_name=block_name,
            block_id=block_id,
        )
    return url


def resolve_dates(
    start: str,
    end: str,
    block_name: str,
    block_id: str,
    *,
    today: date | None = None,
) -> tuple[date, date]:
    """Turn the start and end inputs into dates, counting days in Pacific Time."""
    today = today or pacific_today()
    resolved: list[date] = []
    for name, value in (("start_date", start), ("end_date", end)):
        day = resolve_date(value, today)
        if day is None:
            raise BlockInputError(
                message=(
                    f"{name} '{value}' isn't a date. Use YYYY-MM-DD, today, "
                    "yesterday or NdaysAgo, such as 28daysAgo."
                ),
                block_name=block_name,
                block_id=block_id,
            )
        resolved.append(day)
    first, last = resolved
    if first > last:
        raise BlockInputError(
            message=f"start_date ({first}) is after end_date ({last}).",
            block_name=block_name,
            block_id=block_id,
        )
    return first, last


def match_site(
    domain: str, sites: list[SearchConsoleSite], inspection_url: str = ""
) -> str | None:
    """The first property in ``site_candidates`` order that the account can read."""
    readable = {
        site.site_url.lower(): site.site_url
        for site in sites
        if site.permission_level != UNVERIFIED_LEVEL
    }
    for candidate in site_candidates(domain):
        site_url = readable.get(candidate)
        if site_url and (
            not inspection_url or property_contains(site_url, inspection_url)
        ):
            return site_url
    return None


def as_property(value: str) -> str | None:
    """The property ``value`` names, or None when it is a bare domain.

    URL-prefix properties always end in a slash, so a missing one is added.
    """
    text = value.strip()
    if text.lower().startswith("sc-domain:"):
        return "sc-domain:" + text[len("sc-domain:") :].strip().lower()
    parts = urlsplit(text)
    if parts.scheme in ("http", "https") and parts.netloc:
        path = parts.path if parts.path.endswith("/") else f"{parts.path}/"
        return f"{parts.scheme}://{parts.netloc.lower()}{path}"
    return None


def site_candidates(domain: str) -> list[str]:
    """The properties that can stand for a bare domain, in the order tried.

    Domain properties come first, since they cover every protocol and
    subdomain, then URL-prefix properties, https before http. example.com and
    www.example.com count as one site, with the form given tried first.
    """
    other = (
        domain.removeprefix("www.") if domain.startswith("www.") else f"www.{domain}"
    )
    hosts = (domain, other)
    return [
        *(f"sc-domain:{host}" for host in hosts),
        *(f"https://{host}/" for host in hosts),
        *(f"http://{host}/" for host in hosts),
    ]


def property_contains(site_url: str, url: str) -> bool:
    """Whether the page at ``url`` belongs to the property ``site_url``."""
    page = urlsplit(url)
    if site_url.lower().startswith("sc-domain:"):
        domain = site_url[len("sc-domain:") :].lower()
        host = page.hostname or ""
        return host == domain or host.endswith(f".{domain}")
    prefix = urlsplit(site_url)
    origin = (page.scheme, page.netloc.lower())
    # Hosts ignore case; paths don't.
    if origin != (prefix.scheme, prefix.netloc.lower()):
        return False
    return (page.path or "/").startswith(prefix.path or "/")


def resolve_date(value: str, today: date) -> date | None:
    """Read YYYY-MM-DD, today, yesterday or NdaysAgo; None for anything else."""
    text = value.strip().lower()
    if text == "today":
        return today
    if text == "yesterday":
        return today - timedelta(days=1)
    try:
        if match := _DAYS_AGO.fullmatch(text):
            return today - timedelta(days=int(match.group(1)))
        if _ISO_DATE.fullmatch(text):
            return date.fromisoformat(text)
    except (ValueError, OverflowError):
        pass
    return None


def pacific_today(now: datetime | None = None) -> date:
    """Today's date where Search Console draws its day boundary."""
    moment = now or datetime.now(SEARCH_CONSOLE_TIME_ZONE)
    return moment.astimezone(SEARCH_CONSOLE_TIME_ZONE).date()


def _bare_domain(value: str, block_name: str, block_id: str) -> str:
    domain = value.strip().lower().removesuffix("/")
    if not domain:
        message = (
            "Give the Search Console property to read, such as "
            "sc-domain:example.com or https://www.example.com/, or a domain such "
            "as example.com."
        )
    elif not _DOMAIN.fullmatch(domain):
        message = (
            f"'{value.strip()}' isn't a Search Console property or a domain. Give "
            "the property as Search Console lists it, such as sc-domain:example.com "
            "or https://www.example.com/blog/, or a domain such as example.com."
        )
    else:
        return domain
    raise BlockInputError(message=message, block_name=block_name, block_id=block_id)


def _no_match_message(
    domain: str, sites: list[SearchConsoleSite], inspection_url: str
) -> str:
    readable = [
        site.site_url for site in sites if site.permission_level != UNVERIFIED_LEVEL
    ]
    if not readable:
        return (
            "The connected Google account can't read any Search Console property. "
            "Add it as a user of the property in Search Console (Settings > Users "
            "and permissions), or connect a Google account that has access."
        )
    shown = ", ".join(readable[:MAX_LISTED_SITES])
    if len(readable) > MAX_LISTED_SITES:
        shown += f" and {len(readable) - MAX_LISTED_SITES} more"
    holding = f" that contains {inspection_url}" if inspection_url else ""
    return (
        "None of the Search Console properties the connected Google account can "
        f"read matches {domain}{holding}. It can read: {shown}. Set site_url to "
        "one of these."
    )
