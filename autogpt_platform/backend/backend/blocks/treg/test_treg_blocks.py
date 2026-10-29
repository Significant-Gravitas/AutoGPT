"""Unit tests for the Treg blocks beyond the registry smoke test."""

from typing import Any

import pytest

from backend.blocks.treg._api import TregError, TregResponse, usd_header
from backend.blocks.treg._base import coerce
from backend.blocks.treg._config import (
    PLATFORM_CREDENTIALS_ID,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
)
from backend.blocks.treg._test import mock_miss, mock_response, mock_routed
from backend.blocks.treg.call_tool import TregCallToolBlock
from backend.blocks.treg.catalog import TregGetToolBlock
from backend.blocks.treg.emails import TregFindWorkEmailBlock
from backend.blocks.treg.linkedin import linkedin_reference
from backend.blocks.treg.pages import (
    TregExtractStructuredDataBlock,
    TregExtractWebPageBlock,
)
from backend.blocks.treg.people import TregSearchPeopleBlock
from backend.blocks.treg.phone import TregFindPhoneNumberBlock
from backend.blocks.treg.search import TregSearchSource, TregWebSearchBlock


class FakeTreg:
    """Stands in for the HTTP call and records what each block sent."""

    def __init__(self, response: TregResponse):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, credentials, method, path, params=None, body=None, headers=None
    ):
        self.calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "body": body,
                "headers": headers,
            }
        )
        return self.response


async def run_block(
    block, fake: FakeTreg, credentials=TEST_CREDENTIALS, **inputs
) -> dict[str, Any]:
    block.call_treg = fake
    input_data = block.Input(credentials=TEST_CREDENTIALS_INPUT, **inputs)
    return {
        name: value
        async for name, value in block.run(input_data, credentials=credentials)
    }


async def run_until_error(
    block, fake: FakeTreg, credentials=TEST_CREDENTIALS, **inputs
) -> list[str]:
    """Run the block expecting an error, and return what it emitted first."""
    emitted: list[str] = []
    block.call_treg = fake
    input_data = block.Input(credentials=TEST_CREDENTIALS_INPUT, **inputs)
    with pytest.raises(TregError) as error:
        async for name, _ in block.run(input_data, credentials=credentials):
            emitted.append(name)
    assert emitted == [], f"emitted {emitted} before failing"
    return [str(error.value)]


@pytest.mark.asyncio
async def test_routed_call_sends_only_given_fields_and_the_spending_options():
    fake = FakeTreg(mock_routed({"email": "jane@outerbounds.com"}))
    await run_block(
        TregFindWorkEmailBlock(),
        fake,
        full_name="Jane Doe",
        company_domain="outerbounds.com",
        max_cost_usd=0.2,
        try_more_providers=False,
    )
    assert fake.calls == [
        {
            "method": "POST",
            "path": "call/treg.people.email.find",
            "params": None,
            "body": {"full_name": "Jane Doe", "domain": "outerbounds.com"},
            "headers": {
                "X-Treg-Route-Max-Cost": "0.2",
                "X-Treg-Route-Waterfall": "0",
                "X-Treg-Route-Verify": "true",
            },
        }
    ]


@pytest.mark.asyncio
async def test_a_miss_is_not_found_rather_than_an_error_and_is_still_billed():
    block = TregFindWorkEmailBlock()
    fake = FakeTreg(mock_miss(["email", "verified"], cost_usd=0.012))
    outputs = await run_block(
        block, fake, linkedin_url="https://www.linkedin.com/in/nobody"
    )
    assert outputs == {
        "found": False,
        "providers_tried": ["hunter"],
        "cost_usd": 0.012,
        "call_id": "call_01JTREGTEST0000000000000",
    }
    assert block.execution_stats.provider_cost == 0.012
    assert block.execution_stats.provider_cost_type == "cost_usd"


@pytest.mark.asyncio
async def test_a_failed_call_records_its_charge_before_raising():
    block = TregSearchPeopleBlock()
    fake = FakeTreg(
        mock_response(
            {"detail": {"error": "route_failed", "message": "every candidate failed"}},
            status=502,
            cost_usd=0.03,
        )
    )
    [message] = await run_until_error(block, fake, company_domain="modal.com")
    assert "HTTP 502" in message and "route_failed" in message
    assert block.execution_stats.provider_cost == 0.03


@pytest.mark.asyncio
async def test_a_call_still_pending_after_treg_waited_is_an_error_billed_as_reserved():
    block = TregSearchPeopleBlock()
    fake = FakeTreg(
        mock_response({"_treg": {"outcome": "pending"}}, status=202, cost_usd=0.05)
    )
    [message] = await run_until_error(block, fake, company_domain="modal.com")
    assert "still working" in message
    assert block.execution_stats.provider_cost == 0.05


@pytest.mark.asyncio
async def test_nothing_is_spent_when_no_way_to_name_the_subject_is_given():
    fake = FakeTreg(mock_routed({}))
    # A name without a domain is not enough to find a work email.
    [message] = await run_until_error(TregFindWorkEmailBlock(), fake, full_name="Jane")
    assert "full name and company domain" in message
    assert fake.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verification, output_verified, verified, verdict",
    [
        ({"verdict": "valid", "checked": True}, False, True, "valid"),
        ({"verdict": "catch_all", "checked": True}, True, False, "catch_all"),
        ({"verdict": "unknown", "checked": False}, True, True, None),
        (None, None, False, None),
    ],
)
async def test_an_email_counts_as_verified_only_when_a_check_says_so(
    verification, output_verified, verified, verdict
):
    fake = FakeTreg(
        mock_routed(
            {"email": "jane@outerbounds.com", "verified": output_verified},
            verification=verification,
        )
    )
    outputs = await run_block(
        TregFindWorkEmailBlock(), fake, linkedin_url="https://linkedin.com/in/jane"
    )
    assert outputs["verified"] is verified
    assert outputs.get("verification") == verdict


@pytest.mark.asyncio
async def test_phone_verification_is_off_unless_asked_for():
    fake = FakeTreg(mock_routed({"phone": "+14155550123"}))
    await run_block(TregFindPhoneNumberBlock(), fake, email="jane@outerbounds.com")
    assert "X-Treg-Route-Verify" not in fake.calls[0]["headers"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source, path, body",
    [
        (TregSearchSource.WEB, "call/treg.web.search", {"q": "gpus", "limit": 10}),
        (
            TregSearchSource.GOOGLE,
            "call/treg.google.serp.organic",
            {"q": "gpus", "limit": 10, "country": "gb"},
        ),
        (TregSearchSource.YOUTUBE, "call/treg.youtube.search.videos", {"q": "gpus"}),
        (
            TregSearchSource.NEWS,
            "call/treg.web.search.news",
            {"q": "gpus", "limit": 10},
        ),
        (
            TregSearchSource.GOOGLE_NEWS,
            "call/treg.google.serp.news",
            {"q": "gpus", "limit": 10, "country": "gb"},
        ),
        (
            TregSearchSource.RESEARCH_PAPERS,
            "call/treg.web.search.publications",
            {"q": "gpus", "limit": 10},
        ),
    ],
)
async def test_web_search_routes_each_source(source, path, body):
    field = "videos" if source == TregSearchSource.YOUTUBE else "results"
    fake = FakeTreg(mock_routed({field: [{"title": "A"}, "not-a-row"]}))
    outputs = await run_block(
        TregWebSearchBlock(), fake, query="gpus", source=source, country="gb"
    )
    assert fake.calls[0]["path"] == path
    assert fake.calls[0]["body"] == body
    assert outputs["results"] == [{"title": "A"}]
    assert outputs["count"] == 1


@pytest.mark.asyncio
async def test_structured_extraction_needs_an_instruction_or_a_schema():
    fake = FakeTreg(mock_routed({"data": {}}))
    [message] = await run_until_error(
        TregExtractStructuredDataBlock(), fake, url="https://a.co"
    )
    assert "instruction, a JSON schema" in message
    assert fake.calls == []


@pytest.mark.asyncio
async def test_call_tool_sends_the_request_through_treg():
    fake = FakeTreg(mock_response({"ok": True}, cost_usd=0.002, served_by="x.y"))
    outputs = await run_block(
        TregCallToolBlock(),
        fake,
        tool="dataforseo.backlinks.summary",
        body=[{"target": "modal.com"}],
        headers={"X-Treg-Route-Verify": "true"},
        max_cost_usd=0.1,
    )
    assert fake.calls == [
        {
            "method": "POST",
            "path": "call/dataforseo.backlinks.summary",
            "params": {},
            "body": [{"target": "modal.com"}],
            "headers": {"X-Treg-Route-Verify": "true", "X-Treg-Route-Max-Cost": "0.1"},
        }
    ]
    assert outputs["served_by"] == "x.y"
    assert outputs["cost_usd"] == 0.002


@pytest.mark.asyncio
async def test_call_tool_returns_the_task_of_an_asynchronous_tool():
    task = {"poll": {"endpoint": "minimax.video-gen.query", "param": "task_id"}}
    fake = FakeTreg(
        mock_response({"task_id": "t1"}, status=202, cost_usd=0.4, async_task=task)
    )
    outputs = await run_block(
        TregCallToolBlock(), fake, tool="minimax.video-gen.h3.generate", body={}
    )
    assert outputs["status"] == 202
    assert outputs["async_task"] == task
    assert outputs["cost_usd"] == 0.4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inputs, message",
    [
        ({"tool": "../orgs/1", "body": None}, "is not a Treg tool"),
        ({"tool": "x.y", "body": {"a": 1}, "method": "GET"}, "cannot carry a body"),
    ],
)
async def test_call_tool_refuses_bad_requests_before_sending(inputs, message):
    fake = FakeTreg(mock_response({}))
    [error] = await run_until_error(TregCallToolBlock(), fake, **inputs)
    assert message in error
    assert fake.calls == []


@pytest.mark.asyncio
async def test_call_tool_reports_a_failure_hidden_in_a_success():
    response = mock_response({"tasks": [{"status_code": 40501}]}, cost_usd=0.001)
    response.upstream_failed = True
    block = TregCallToolBlock()
    [message] = await run_until_error(block, FakeTreg(response), tool="dataforseo.x")
    assert "reported a failure" in message
    assert block.execution_stats.provider_cost == 0.001


@pytest.mark.asyncio
async def test_get_tool_rejects_ids_that_are_not_tool_ids():
    fake = FakeTreg(mock_response({}))
    [message] = await run_until_error(TregGetToolBlock(), fake, tool_id="a/../b")
    assert "is not a Treg tool id" in message
    assert fake.calls == []


@pytest.mark.parametrize(
    "value, kind, expected",
    [
        ("1,200", int, 1200),
        ("500+", int, None),
        (3.0, int, 3),
        (True, int, None),
        (0.5, float, 0.5),
        ("0.5", float, None),
        ("", str, None),
        (2021, str, None),
        ({"city": "Paris"}, str, None),
        ("yes", bool, None),
        (False, bool, False),
    ],
)
def test_coerce_keeps_only_values_of_the_declared_type(value, kind, expected):
    assert coerce(value, kind) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header",
    ["X-Treg-Route-Max-Cost", "x-treg-route-max-cost", "X-TREG-ORG", "x-treg-meta"],
)
async def test_call_tool_refuses_treg_headers_that_are_not_the_callers(header):
    # A second spending-cap header in another case would reach Treg first and
    # lift the cap; team and billing headers are Treg's own.
    fake = FakeTreg(mock_response({}))
    [message] = await run_until_error(
        TregCallToolBlock(), fake, tool="x.y", headers={header: "500"}
    )
    assert "cannot be set here" in message
    assert fake.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool", ["intercom/conversations", "https://api.stripe.com/v1/charges"]
)
async def test_the_platform_key_cannot_reach_the_platform_teams_own_tools(tool):
    fake = FakeTreg(mock_response({}))
    [message] = await run_until_error(
        TregCallToolBlock(), fake, credentials=PLATFORM_KEY, tool=tool
    )
    assert "is a team tool" in message
    assert fake.calls == []


@pytest.mark.asyncio
async def test_the_platform_key_can_call_catalog_tools():
    fake = FakeTreg(mock_response({"ok": True}))
    outputs = await run_block(
        TregCallToolBlock(),
        fake,
        credentials=PLATFORM_KEY,
        tool="tikhub.tiktok.user.profile",
    )
    assert outputs["response"] == {"ok": True}


PLATFORM_KEY = TEST_CREDENTIALS.model_copy(update={"id": PLATFORM_CREDENTIALS_ID})


@pytest.mark.parametrize(
    "reference, expected",
    [
        (
            "https://www.linkedin.com/in/janedoe",
            {"linkedin_url": "https://www.linkedin.com/in/janedoe"},
        ),
        (
            "linkedin.com/company/modal-labs",
            {"linkedin_url": "https://linkedin.com/company/modal-labs"},
        ),
        (
            "linkedin.com/in/jane?ref=https://example.com",
            {"linkedin_url": "https://linkedin.com/in/jane?ref=https://example.com"},
        ),
        (
            "HTTPS://www.linkedin.com/in/jane",
            {"linkedin_url": "https://www.linkedin.com/in/jane"},
        ),
        (" janedoe ", {"linkedin_handle": "janedoe"}),
        # Not LinkedIn, whatever its path says.
        (
            "https://evil.example/linkedin.com/in/x",
            {"linkedin_handle": "https://evil.example/linkedin.com/in/x"},
        ),
        (
            "https://linkedin.com.evil.example/in/x",
            {"linkedin_handle": "https://linkedin.com.evil.example/in/x"},
        ),
    ],
)
def test_linkedin_reference_sends_only_linkedin_hosts_as_urls(reference, expected):
    assert linkedin_reference(reference) == expected


@pytest.mark.asyncio
async def test_an_email_that_is_not_a_string_is_not_found():
    fake = FakeTreg(mock_routed({"email": {"value": "x"}}))
    outputs = await run_block(
        TregFindWorkEmailBlock(), fake, linkedin_url="https://linkedin.com/in/jane"
    )
    assert "email" not in outputs
    assert outputs["found"] is False


@pytest.mark.asyncio
async def test_a_page_with_only_html_has_no_content_output():
    page = {"url": "https://a.co", "html": "<p>Hi</p>"}
    fake = FakeTreg(mock_routed({"pages": [page]}))
    outputs = await run_block(TregExtractWebPageBlock(), fake, url="https://a.co")
    assert "content" not in outputs
    assert outputs["pages"] == [page]


@pytest.mark.asyncio
async def test_a_work_email_miss_emits_no_name_fields():
    fake = FakeTreg(mock_routed({"email": None, "first_name": "Jane"}))
    outputs = await run_block(
        TregFindWorkEmailBlock(), fake, linkedin_url="https://linkedin.com/in/jane"
    )
    assert "first_name" not in outputs
    assert outputs["found"] is False


@pytest.mark.parametrize(
    "usd, header", [(0.5, "0.5"), (25.0, "25"), (0.00005, "0.00005"), (0.0, "0")]
)
def test_spending_caps_are_sent_as_plain_decimals(usd, header):
    assert usd_header(usd) == header


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verification, verdict",
    [
        ({"checked": True, "verdict": "live"}, "live"),
        ({"checked": False, "verdict": "unknown", "reason": "not_international"}, None),
    ],
)
async def test_phone_verification_is_reported_only_when_the_check_ran(
    verification, verdict
):
    fake = FakeTreg(mock_routed({"phone": "+14155550123"}, verification=verification))
    outputs = await run_block(
        TregFindPhoneNumberBlock(), fake, email="jane@outerbounds.com", verify=True
    )
    assert fake.calls[0]["headers"]["X-Treg-Route-Verify"] == "true"
    assert outputs.get("verification") == verdict
