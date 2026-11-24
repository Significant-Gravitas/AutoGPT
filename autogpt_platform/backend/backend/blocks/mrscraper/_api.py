"""Thin async client for the MrScraper APIs (https://mrscraper.com).

Status calls the platform API; the six scraping and search services call the
Scraper API with query parameters. Each request builder sends its fields in
the documented order, and ``send`` makes one attempt: a retried scrape can be
billed again, and MrScraper retries a blocked page itself (``maxRetries``).
"""

import re
from collections.abc import Sequence
from typing import Any, Literal
from urllib.parse import urlencode, urlsplit

from aiohttp import ClientError, ClientTimeout

from backend.data.execution import ExecutionContext
from backend.sdk import (
    APIKeyCredentials,
    BaseModel,
    MediaFileType,
    Requests,
    SchemaField,
    store_media_file,
)
from backend.util.request import Response

SCRAPER_API_URL = "https://api.mrscraper.com/"
ACCOUNT_STATUS_URL = "https://api.app.mrscraper.com/api/v1/subscription-accounts"

MAX_GET_URL_LENGTH = 8000
STATUS_DEADLINE_SECONDS = 600
FIXED_DEADLINE_SECONDS = 120
SCRAPE_DEADLINE_MARGIN_SECONDS = 30

DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 330
DEFAULT_PROMPT = "Extract all data as complete as possible."

_VIEWPORT = re.compile(r"^(\d{3,4})x(\d{3,4})(?:-(\d{2,3}))?$")
_COUNTRY_CODE = re.compile(r"^[a-z]{2}$")


class MrScraperError(ValueError):
    """A failed MrScraper call. ValueError puts it on the block's error output."""


class MrScraperRequest(BaseModel):
    """One upstream request: method, URL, ordered query fields and deadline."""

    method: Literal["GET", "POST"] = "GET"
    url: str = SCRAPER_API_URL
    query: list[tuple[str, str]]
    body: dict[str, str] | None = None
    deadline_seconds: int


class MrScraperSearchResult(BaseModel):
    """One ranked web search result."""

    rank: int | None = SchemaField(description="Position of the result", default=None)
    title: str = SchemaField(description="Title of the result page", default="")
    url: str = SchemaField(description="URL of the result page", default="")
    description: str = SchemaField(
        description="Description of the result page", default=""
    )
    source: str = SchemaField(description="Source site of the result", default="")


def url_field() -> Any:
    return SchemaField(
        description="Absolute HTTP or HTTPS URL of the page to process.",
        placeholder="https://example.com",
    )


def browser_rendering_field() -> Any:
    return SchemaField(
        description="Whether to load the page in a browser: true always does and "
        "false never does. Omit it to load the page without a browser first and "
        "switch to a browser automatically if the request is blocked.",
        default=None,
    )


def timeout_field() -> Any:
    return SchemaField(
        description="Maximum seconds to wait for the scrape, from 1 to 330. The "
        "HTTP request allows 30 additional seconds.",
        default=DEFAULT_TIMEOUT_SECONDS,
        ge=1,
        le=MAX_TIMEOUT_SECONDS,
    )


def warm_start_field() -> Any:
    return SchemaField(
        description="Whether to open the site's home page first to warm the "
        "session before loading the target URL.",
        default=False,
    )


def block_resources_field() -> Any:
    return SchemaField(
        description="Whether to block images, fonts, and stylesheets while "
        "loading the page.",
        default=False,
    )


def wait_for_selector_field() -> Any:
    return SchemaField(
        description="CSS selector or text to wait for on the page before "
        "returning the result, up to 500 characters.",
        default=None,
        max_length=500,
    )


def action_field() -> Any:
    return SchemaField(
        description="Scraper action that routes the request or simply flags it, "
        "useful for separating analytics per domain.",
        default=None,
    )


def geo_code_field() -> Any:
    return SchemaField(
        description="Two-letter country code to load the page from, sent in "
        "lowercase.",
        default=None,
    )


def max_age_field() -> Any:
    return SchemaField(
        description="Return a cached result of the same request when it is newer "
        "than the selected age. A cached result costs the same tokens as the "
        "original scrape.",
        default=0,
        ge=0,
    )


def max_retries_field() -> Any:
    return SchemaField(
        description="Maximum number of retries, as a nonnegative integer. Omit "
        "this parameter for no limit.",
        default=None,
        ge=0,
    )


def page_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise MrScraperError(
            f"'{url}' is not an absolute HTTP or HTTPS URL, such as "
            "https://example.com."
        )
    return url.strip()


def country_code(value: str | None, field: str) -> str | None:
    if value is None or not value.strip():
        return None
    code = value.strip().lower()
    if not _COUNTRY_CODE.match(code):
        raise MrScraperError(
            f"{field} must be a two-letter country code, such as us or gb."
        )
    return code


def screenshot_area(value: str) -> str:
    """full, top, or a WIDTHxHEIGHT[-ZOOM] viewport, in lowercase."""
    area = value.strip().lower()
    if area in ("full", "top"):
        return area
    match = _VIEWPORT.match(area)
    if match:
        width, height, zoom = match.groups()
        if (
            100 <= int(width) <= 3840
            and 100 <= int(height) <= 3840
            and (zoom is None or 25 <= int(zoom) <= 300)
        ):
            return area
    raise MrScraperError(
        f"'{value}' is not a screenshot area. Use full, top, or a WIDTHxHEIGHT "
        "viewport from 100 to 3840 pixels with an optional -ZOOM percent from 25 "
        "to 300, such as 1280x800 or 1920x1080-75."
    )


def _query(
    fields: Sequence[tuple[str, str | int | bool | None]]
) -> list[tuple[str, str]]:
    """Drop unset fields; booleans are sent as true or false."""
    return [
        (name, str(value).lower() if isinstance(value, bool) else str(value))
        for name, value in fields
        if value is not None
    ]


def _scrape_deadline(timeout: int) -> int:
    return timeout + SCRAPE_DEADLINE_MARGIN_SECONDS


def status_request() -> MrScraperRequest:
    return MrScraperRequest(
        url=ACCOUNT_STATUS_URL, query=[], deadline_seconds=STATUS_DEADLINE_SECONDS
    )


def scrape_html_request(
    *,
    url: str,
    browser_rendering: bool | None,
    timeout: int,
    warm_start: bool,
    block_resources: bool,
    wait_for_selector: str | None,
    action: str | None,
    geo_code: str | None,
    max_age: int,
    max_retries: int | None,
    screenshot: str,
) -> MrScraperRequest:
    area = screenshot.strip().lower() or "false"
    if area != "false":
        area = screenshot_area(area)
        browser_rendering = True
    return MrScraperRequest(
        query=_query(
            [
                ("url", page_url(url)),
                ("html", True),
                ("markdown", False),
                ("browserRendering", browser_rendering),
                ("timeout", timeout),
                ("warmStart", warm_start),
                ("blockResources", block_resources),
                ("waitForSelector", wait_for_selector or None),
                ("action", action or None),
                ("geoCode", country_code(geo_code, "geo_code")),
                ("saveResult", True),
                ("maxAge", max_age),
                ("maxRetries", max_retries),
                ("screenshot", area),
            ]
        ),
        deadline_seconds=_scrape_deadline(timeout),
    )


def scrape_markdown_request(
    *,
    url: str,
    browser_rendering: bool | None,
    timeout: int,
    warm_start: bool,
    block_resources: bool,
    wait_for_selector: str | None,
    action: str | None,
    geo_code: str | None,
    max_age: int,
    max_retries: int | None,
) -> MrScraperRequest:
    return MrScraperRequest(
        query=_query(
            [
                ("url", page_url(url)),
                ("markdown", True),
                ("html", False),
                ("screenshot", False),
                ("browserRendering", browser_rendering),
                ("timeout", timeout),
                ("warmStart", warm_start),
                ("blockResources", block_resources),
                ("waitForSelector", wait_for_selector or None),
                ("action", action or None),
                ("geoCode", country_code(geo_code, "geo_code")),
                ("saveResult", True),
                ("maxAge", max_age),
                ("maxRetries", max_retries),
            ]
        ),
        deadline_seconds=_scrape_deadline(timeout),
    )


def screenshot_request(
    *,
    url: str,
    screenshot: str,
    timeout: int,
    wait_for_selector: str | None,
    geo_code: str | None,
    max_age: int,
    max_retries: int | None,
) -> MrScraperRequest:
    return MrScraperRequest(
        query=_query(
            [
                ("url", page_url(url)),
                ("screenshot", screenshot_area(screenshot)),
                ("html", False),
                ("markdown", False),
                ("browserRendering", True),
                ("timeout", timeout),
                ("waitForSelector", wait_for_selector or None),
                ("geoCode", country_code(geo_code, "geo_code")),
                ("saveResult", True),
                ("maxAge", max_age),
                ("maxRetries", max_retries),
            ]
        ),
        deadline_seconds=_scrape_deadline(timeout),
    )


def scrape_brand_request(*, url: str) -> MrScraperRequest:
    return MrScraperRequest(
        query=_query(
            [
                ("action", "brand-scraping"),
                ("url", page_url(url)),
                ("bundle", "none"),
            ]
        ),
        deadline_seconds=FIXED_DEADLINE_SECONDS,
    )


def scrape_json_request(
    *,
    url: str,
    prompt: str,
    html: bool,
    browser_rendering: bool | None,
    timeout: int,
    warm_start: bool,
    block_resources: bool,
    wait_for_selector: str | None,
    geo_code: str | None,
    max_age: int,
    max_retries: int | None,
    screenshot: bool,
) -> MrScraperRequest:
    target = [
        ("url", page_url(url)),
        ("prompt", prompt.strip() or DEFAULT_PROMPT),
        ("agent", "general"),
    ]
    options = [
        ("html", html),
        ("browserRendering", browser_rendering),
        ("timeout", timeout),
        ("warmStart", warm_start),
        ("blockResources", block_resources),
        ("waitForSelector", wait_for_selector or None),
        ("geoCode", country_code(geo_code, "geo_code")),
        ("saveResult", True),
        ("maxAge", max_age),
        ("maxRetries", max_retries),
        ("screenshot", screenshot),
    ]
    query = _query([*target, *options])
    if len(f"{SCRAPER_API_URL}?{urlencode(query)}") <= MAX_GET_URL_LENGTH:
        return MrScraperRequest(query=query, deadline_seconds=_scrape_deadline(timeout))
    return MrScraperRequest(
        method="POST",
        query=_query(options),
        body=dict(_query(target)),
        deadline_seconds=_scrape_deadline(timeout),
    )


def web_search_request(
    *, query: str, country: str | None, num_results: int
) -> MrScraperRequest:
    text = query.strip()
    if not 1 <= len(text) <= 1000:
        raise MrScraperError("query must contain 1 to 1000 characters.")
    return MrScraperRequest(
        query=_query(
            [
                ("action", "websearch"),
                ("query", text),
                ("country", country_code(country, "country")),
                ("numResults", num_results),
            ]
        ),
        deadline_seconds=FIXED_DEADLINE_SECONDS,
    )


async def send(credentials: APIKeyCredentials, request: MrScraperRequest) -> Any:
    """Send one request and return the decoded body (JSON or text)."""
    api_key = credentials.api_key.get_secret_value()
    requests = Requests(
        trusted_origins=[request.url],
        raise_for_status=False,
        retry_max_attempts=1,
        extra_headers={"x-api-token": api_key, "Authorization": f"Bearer {api_key}"},
        sensitive_headers=["x-api-token"],
    )
    try:
        response = await requests.request(
            request.method,
            request.url,
            params=request.query or None,
            json=request.body,
            allow_redirects=False,
            timeout=ClientTimeout(total=request.deadline_seconds),
        )
    except TimeoutError as e:
        raise MrScraperError(
            f"MrScraper did not answer within {request.deadline_seconds} seconds."
        ) from e
    except ClientError as e:
        raise MrScraperError(f"Could not reach MrScraper: {type(e).__name__}") from e
    body = _decode(response)
    if not response.ok:
        raise MrScraperError(
            f"MrScraper returned HTTP {response.status}: {error_message(body)}"
        )
    return body


def _decode(response: Response) -> Any:
    if "json" in response.headers.get("content-type", "").lower():
        try:
            return response.json()
        except ValueError:
            pass
    return response.text()


def error_message(body: Any) -> str:
    if isinstance(body, dict):
        parts = [
            str(body[key])
            for key in ("message", "data", "error", "errors")
            if isinstance(body.get(key), str) and body[key]
        ]
        parts += _validation_errors(body.get("detail"))
        if parts:
            return ": ".join(dict.fromkeys(parts))
    text = str(body).strip()
    return text[:500] if text else "no details"


def _validation_errors(detail: Any) -> list[str]:
    """Field errors of a 422 answer, such as "prompt: String should have at most
    8000 characters"."""
    if isinstance(detail, str):
        return [detail]
    if not isinstance(detail, list):
        return []
    errors = []
    for item in detail:
        if isinstance(item, dict) and isinstance(item.get("msg"), str):
            field = ".".join(
                str(part) for part in item.get("loc") or [] if part != "body"
            )
            errors.append(f"{field}: {item['msg']}" if field else item["msg"])
    return errors


def png_data_uri(screenshot: str) -> str:
    """The API returns a bare base64 PNG; make it a data URI the platform stores."""
    if screenshot.startswith("data:"):
        return screenshot
    return f"data:image/png;base64,{screenshot}"


async def store_screenshot(
    screenshot: str, execution_context: ExecutionContext
) -> MediaFileType:
    return await store_media_file(
        file=MediaFileType(png_data_uri(screenshot)),
        execution_context=execution_context,
        return_format="for_block_output",
    )


def text_field(body: Any, key: str) -> str:
    """A string field of a JSON body, or a plain-text body itself."""
    value = body.get(key) if isinstance(body, dict) else body
    if not isinstance(value, str):
        raise MrScraperError(f"MrScraper returned no {key} for this page.")
    return value


def json_object(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise MrScraperError(
            f"MrScraper returned an unexpected response: {error_message(body)}"
        )
    return body
