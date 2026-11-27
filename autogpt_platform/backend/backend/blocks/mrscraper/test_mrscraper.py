"""MrScraper requests match the Standard Service Reference examples."""

from typing import Any
from urllib.parse import urlencode

import pytest
from pydantic import SecretStr

from backend.blocks.mrscraper import _api
from backend.blocks.mrscraper._api import (
    ACCOUNT_STATUS_URL,
    SCRAPER_API_URL,
    MrScraperError,
    scrape_brand_request,
    scrape_html_request,
    scrape_json_request,
    scrape_markdown_request,
    screenshot_request,
    send,
    status_request,
    web_search_request,
)
from backend.data.model import APIKeyCredentials

CREDENTIALS = APIKeyCredentials(
    provider="mrscraper", api_key=SecretStr("atk_test"), title="Test key"
)

PAGE_DEFAULTS: dict[str, Any] = {
    "url": "https://example.com",
    "browser_rendering": None,
    "timeout": 300,
    "warm_start": False,
    "block_resources": False,
    "wait_for_selector": None,
    "geo_code": None,
    "max_age": 0,
    "max_retries": None,
}

JSON_SCHEMA_PROMPT = (
    "Extract the first 2 items with title and price. Json Schema: "
    '{"type":"array","items":{"type":"object","properties":{"title":{"type":'
    '"string"},"price":{"type":"string"}},"required":["title","price"]}}'
)


def html_request(**overrides: Any) -> _api.MrScraperRequest:
    return scrape_html_request(
        **{**PAGE_DEFAULTS, "action": None, "screenshot": "false", **overrides}
    )


def json_request(**overrides: Any) -> _api.MrScraperRequest:
    return scrape_json_request(
        **{
            **PAGE_DEFAULTS,
            "prompt": _api.DEFAULT_PROMPT,
            "html": True,
            "screenshot": False,
            **overrides,
        }
    )


def test_status_has_no_query_and_a_fixed_600_second_deadline():
    request = status_request()
    assert (request.method, request.url, request.query) == (
        "GET",
        ACCOUNT_STATUS_URL,
        [],
    )
    assert request.deadline_seconds == 600


def test_scrape_html_example():
    request = html_request(
        browser_rendering=True, warm_start=True, wait_for_selector="body", timeout=120
    )
    assert request.url == SCRAPER_API_URL
    assert request.query == [
        ("url", "https://example.com"),
        ("html", "true"),
        ("markdown", "false"),
        ("browserRendering", "true"),
        ("timeout", "120"),
        ("warmStart", "true"),
        ("blockResources", "false"),
        ("waitForSelector", "body"),
        ("saveResult", "true"),
        ("maxAge", "0"),
        ("screenshot", "false"),
    ]
    assert request.deadline_seconds == 150


def test_scrape_html_screenshot_forces_browser_rendering():
    request = dict(
        html_request(browser_rendering=False, screenshot="1920X1080-75").query
    )
    assert request["browserRendering"] == "true"
    assert request["screenshot"] == "1920x1080-75"


@pytest.mark.parametrize("area", ["99x800", "1280x3841", "1280x800-301", "side"])
def test_scrape_html_rejects_an_invalid_screenshot_before_sending(area: str):
    with pytest.raises(MrScraperError, match="not a screenshot area"):
        html_request(screenshot=area)


def test_scrape_markdown_example():
    request = scrape_markdown_request(
        **{**PAGE_DEFAULTS, "action": None, "warm_start": True, "max_age": 86400000}
    )
    assert request.query == [
        ("url", "https://example.com"),
        ("markdown", "true"),
        ("html", "false"),
        ("screenshot", "false"),
        ("timeout", "300"),
        ("warmStart", "true"),
        ("blockResources", "false"),
        ("saveResult", "true"),
        ("maxAge", "86400000"),
    ]
    assert request.deadline_seconds == 330


def test_get_screenshot_example():
    request = screenshot_request(
        url="https://example.com",
        screenshot="top",
        timeout=120,
        wait_for_selector=None,
        geo_code=None,
        max_age=0,
        max_retries=None,
    )
    assert request.query == [
        ("url", "https://example.com"),
        ("screenshot", "top"),
        ("html", "false"),
        ("markdown", "false"),
        ("browserRendering", "true"),
        ("timeout", "120"),
        ("saveResult", "true"),
        ("maxAge", "0"),
    ]


def test_scrape_brand_example():
    request = scrape_brand_request(url="https://example.com")
    assert urlencode(request.query) == (
        "action=brand-scraping&url=https%3A%2F%2Fexample.com&bundle=none"
    )
    assert request.deadline_seconds == 120


def test_scrape_json_example_is_a_get_with_the_encoded_prompt():
    request = json_request(prompt=JSON_SCHEMA_PROMPT, browser_rendering=True)
    assert request.method == "GET"
    assert urlencode(request.query) == (
        "url=https%3A%2F%2Fexample.com"
        "&prompt=Extract+the+first+2+items+with+title+and+price.+Json+Schema%3A+"
        "%7B%22type%22%3A%22array%22%2C%22items%22%3A%7B%22type%22%3A%22object%22"
        "%2C%22properties%22%3A%7B%22title%22%3A%7B%22type%22%3A%22string%22%7D"
        "%2C%22price%22%3A%7B%22type%22%3A%22string%22%7D%7D%2C%22required%22%3A"
        "%5B%22title%22%2C%22price%22%5D%7D%7D"
        "&agent=general&html=true&browserRendering=true&timeout=300"
        "&warmStart=false&blockResources=false&saveResult=true&maxAge=0"
        "&screenshot=false"
    )


def test_scrape_json_moves_target_fields_to_a_post_body_past_8000_characters():
    prompt = "x" * 8000
    request = json_request(prompt=prompt, geo_code="GB")
    assert request.method == "POST"
    assert request.body == {
        "url": "https://example.com",
        "prompt": prompt,
        "agent": "general",
    }
    assert [name for name, _ in request.query] == [
        "html",
        "timeout",
        "warmStart",
        "blockResources",
        "geoCode",
        "saveResult",
        "maxAge",
        "screenshot",
    ]
    assert dict(request.query)["geoCode"] == "gb"


def test_web_search_example():
    request = web_search_request(
        query="  site:example.com documentation ", country="ID", num_results=20
    )
    assert urlencode(request.query) == (
        "action=websearch&query=site%3Aexample.com+documentation"
        "&country=id&numResults=20"
    )
    assert request.deadline_seconds == 120


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: web_search_request(query="   ", country=None, num_results=10),
            "1 to 1000",
        ),
        (lambda: html_request(url="example.com"), "absolute HTTP or HTTPS URL"),
        (lambda: html_request(geo_code="usa"), "two-letter country code"),
    ],
)
def test_invalid_input_fails_before_sending(build, message: str):
    with pytest.raises(MrScraperError, match=message):
        build()


class FakeResponse:
    def __init__(self, status: int, body: str, content_type: str):
        self.status = status
        self.headers = {"content-type": content_type}
        self._body = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        import json

        return json.loads(self._body)

    def text(self) -> str:
        return self._body


class FakeRequests:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.init: dict[str, Any] = {}
        self.sent: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, **kwargs: Any) -> "FakeRequests":
        self.init = kwargs
        return self

    async def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.sent.append((method, url, kwargs))
        return self.response


@pytest.fixture
def fake_requests(monkeypatch: pytest.MonkeyPatch):
    def install(status: int, body: str, content_type: str) -> FakeRequests:
        fake = FakeRequests(FakeResponse(status, body, content_type))
        monkeypatch.setattr(_api, "Requests", fake)
        return fake

    return install


async def test_send_authenticates_by_header_once_without_redirects(fake_requests):
    fake = fake_requests(200, "# Example page", "text/markdown; charset=utf-8")
    request = scrape_markdown_request(**{**PAGE_DEFAULTS, "action": None})

    body = await send(CREDENTIALS, request)

    assert body == "# Example page"
    api_key = CREDENTIALS.api_key.get_secret_value()
    assert fake.init["extra_headers"] == {
        "x-api-token": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    assert fake.init["retry_max_attempts"] == 1
    assert fake.init["raise_for_status"] is False
    [(method, url, kwargs)] = fake.sent
    assert (method, url) == ("GET", SCRAPER_API_URL)
    assert api_key not in str(kwargs["params"])
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"].total == 330


async def test_send_posts_the_json_body_for_long_prompts(fake_requests):
    fake = fake_requests(200, '{"success": true, "data": {}}', "application/json")
    request = json_request(prompt="x" * 8000)

    assert await send(CREDENTIALS, request) == {
        "success": True,
        "data": {},
    }
    [(method, _, kwargs)] = fake.sent
    assert method == "POST"
    assert kwargs["json"]["agent"] == "general"


@pytest.mark.parametrize(
    "status, body, message",
    [
        (
            401,
            '{"message":"Unauthorized access","data":null}',
            "HTTP 401: Unauthorized access",
        ),
        (
            400,
            '{"success":false,"message":"Invalid URL","data":"url must be an '
            'encoded http or https URL","error":"bad_request"}',
            "HTTP 400: Invalid URL: url must be an encoded http or https URL: "
            "bad_request",
        ),
        (
            422,
            '{"detail":[{"type":"string_too_long","loc":["body","prompt"],'
            '"msg":"String should have at most 8000 characters"}]}',
            "HTTP 422: prompt: String should have at most 8000 characters",
        ),
    ],
)
async def test_send_reports_the_api_error_message(
    fake_requests, status: int, body: str, message: str
):
    fake_requests(status, body, "application/json; charset=utf-8")

    with pytest.raises(MrScraperError) as error:
        await send(CREDENTIALS, status_request())

    assert str(error.value) == f"MrScraper returned {message}"
