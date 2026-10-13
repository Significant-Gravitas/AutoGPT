import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.context import (
    MAX_OPENUI_RESEARCH_CALLS,
    enable_openui_research_budget,
    set_execution_context,
    take_web_research_slot,
    web_research_progress,
)
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import ErrorResponse, ResponseType
from backend.copilot.tools.web_fetch import WebFetchTool
from backend.copilot.tools.web_search import WebSearchTool


@pytest.fixture(autouse=True)
def fresh_turn():
    set_execution_context("owner", make_session("owner"))
    yield
    set_execution_context(None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ink, expected", [("null", "Unknown / incomplete inputs"), ("18.67", "$314.04")]
)
async def test_saved_summary_uses_the_rendered_calculation(ink, expected):
    source = (
        'root = Workspace("Printer", "Supplied quote", [form, total], "printer")\n'
        'form = Form("quote", "Price", [NumberField("ink", "Monthly ink", '
        f'{ink}, 0, null, 0.01)], "Compare", "Compare quote")\n'
        'total = CalculatedMetric("quote", "Annual cost", "90+12*ink", "currency", "USD", 2)'
    )
    result = await TOOL_REGISTRY["render_ui"]._execute(
        "owner",
        make_session("owner"),
        source=source,
        summary="Annual cost is $90. It ties at exactly $314.",
    )
    assert result.type == ResponseType.UI_RENDERED
    assert f"Annual cost: {expected}" in result.message
    assert "Annual cost is $90" not in result.message
    assert "ties" not in result.message


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 403, 404, 500])
async def test_error_page_footer_cannot_be_used_as_verified_evidence(status):
    response = MagicMock()
    response.status = status
    response.content = b"<h1>Not found</h1><footer>Open daily, 9 to 5</footer>"
    with patch("backend.copilot.tools.web_fetch.Requests") as requests:
        requests.return_value.get = AsyncMock(return_value=response)
        result = await WebFetchTool()._execute(
            "owner", make_session("owner"), url="https://example.com/hours"
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "http_error"
    assert str(status) in result.message
    assert "Open daily" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_parallel_research_calls_share_one_allowance_and_cannot_reset_it():
    enable_openui_research_budget()

    async def lookup():
        await asyncio.sleep(0)
        return take_web_research_slot()

    results = await asyncio.gather(
        *(lookup() for _ in range(MAX_OPENUI_RESEARCH_CALLS + 4))
    )
    assert results.count(None) == MAX_OPENUI_RESEARCH_CALLS
    assert len([result for result in results if result]) == 4
    enable_openui_research_budget()
    assert take_web_research_slot() is not None
    set_execution_context("owner", make_session("owner"))
    assert all(
        take_web_research_slot() is None for _ in range(MAX_OPENUI_RESEARCH_CALLS + 2)
    )


@pytest.mark.asyncio
async def test_budget_counts_prior_lookups_and_blocks_network_at_both_tools():
    for _ in range(MAX_OPENUI_RESEARCH_CALLS):
        assert take_web_research_slot() is None
    enable_openui_research_budget()
    with patch("backend.copilot.tools.web_fetch.Requests") as fetch, patch(
        "backend.copilot.tools.web_search.AsyncOpenAI"
    ) as search:
        session = make_session("owner")
        fetched = await WebFetchTool()._execute(
            "owner", session, url="https://example.com"
        )
        searched = await WebSearchTool()._execute(
            "owner", session, query="museum hours"
        )
        assert "limit" in fetched.message
        assert "limit" in searched.message
        fetch.assert_not_called()
        search.assert_not_called()


@pytest.mark.asyncio
async def test_turns_have_isolated_research_allowances():
    async def run_turn():
        set_execution_context("owner", make_session("owner"))
        enable_openui_research_budget()
        await asyncio.sleep(0)
        return [take_web_research_slot() for _ in range(MAX_OPENUI_RESEARCH_CALLS + 1)]

    first, second = await asyncio.gather(run_turn(), run_turn())
    assert first.count(None) == second.count(None) == MAX_OPENUI_RESEARCH_CALLS


@pytest.mark.asyncio
async def test_successful_fetch_reports_remaining_research_only_for_interactive_turns():
    response = MagicMock()
    response.status = 200
    response.url = "https://example.com/"
    response.headers = {"content-type": "text/plain"}
    response.content = b"Opening hours have not been supplied."
    assert web_research_progress() == ""
    with patch("backend.copilot.tools.web_fetch.Requests") as requests:
        requests.return_value.get = AsyncMock(return_value=response)
        tool = WebFetchTool()
        first = await tool._execute("owner", make_session("owner"), url=response.url)
        assert first.message == "Fetched https://example.com/"
        enable_openui_research_budget()
        second = await tool._execute("owner", make_session("owner"), url=response.url)
    assert f"2 of {MAX_OPENUI_RESEARCH_CALLS} lookups used" in second.message
    assert "required destinations" in second.message
