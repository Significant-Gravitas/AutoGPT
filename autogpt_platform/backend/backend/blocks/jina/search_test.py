import json
from urllib.parse import quote

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from backend.blocks.helpers import http
from backend.blocks.jina._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.jina.search import SearchTheWebBlock
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.request import HTTPClientError, HTTPServerError, Requests

QUERY = 'PES 2013 "no search results" & 中文?#'
NO_RESULTS_MESSAGE = f"No search results available for query {QUERY}"


@pytest_asyncio.fixture
async def jina_http_response(request: pytest.FixtureRequest, monkeypatch):
    status, body = request.param
    received: list[str] = []

    async def respond(incoming: web.Request):
        received.append(incoming.path)
        assert incoming.headers["Authorization"] == (
            f"Bearer {TEST_CREDENTIALS.api_key.get_secret_value()}"
        )
        return web.Response(status=status, text=body)

    app = web.Application()
    app.router.add_get("/", respond)
    async with TestServer(app) as server:
        url = str(server.make_url("/"))

        class LocalJinaRequests(Requests):
            def __init__(self):
                super().__init__(trusted_origins=[url], retry_max_attempts=1)

            async def get(self, requested_url: str, *args, **kwargs):
                assert requested_url == f"https://s.jina.ai/{quote(QUERY)}"
                return await super().get(url, *args, **kwargs)

        monkeypatch.setattr(http, "Requests", LocalJinaRequests)
        yield received


@pytest.mark.parametrize(
    "jina_http_response",
    [
        (
            422,
            json.dumps(
                {
                    "data": None,
                    "code": 422,
                    "name": "AssertionFailureError",
                    "status": 42206,
                    "message": NO_RESULTS_MESSAGE,
                }
            ),
        ),
        (422, json.dumps({"message": "No search results available"})),
        (422, json.dumps({"message": "  NO  SEARCH\nRESULTS available for query x  "})),
        (422, json.dumps({"detail": NO_RESULTS_MESSAGE})),
        (422, json.dumps({"error": NO_RESULTS_MESSAGE})),
        (422, NO_RESULTS_MESSAGE),
    ],
    indirect=True,
)
async def test_no_results_is_empty_through_http_and_block_executor(jina_http_response):
    block = SearchTheWebBlock()

    outputs = [
        output
        async for output in block.execute(
            {"query": QUERY, "credentials": TEST_CREDENTIALS_INPUT},
            execution_context=ExecutionContext(),
            credentials=TEST_CREDENTIALS,
        )
    ]

    assert outputs == [("results", "")]
    assert not outputs[0][1]
    assert block.execution_stats.provider_cost is None
    assert block.execution_stats.provider_cost_type is None
    assert jina_http_response == ["/"]


@pytest.mark.parametrize(
    "jina_http_response",
    [
        (422, json.dumps({"message": "Validation error"})),
        (422, json.dumps({"message": None})),
        (422, json.dumps({"message": 42206})),
        (422, json.dumps({"message": False})),
        (422, json.dumps({"message": [NO_RESULTS_MESSAGE]})),
        (422, json.dumps({"message": {"text": NO_RESULTS_MESSAGE}})),
        (422, json.dumps({"query": NO_RESULTS_MESSAGE, "message": "Invalid query"})),
        (422, "null"),
        (422, "false"),
        (422, "42"),
        (422, "{}"),
        (422, "not-json"),
        (422, json.dumps({"message": f"Invalid query: {NO_RESULTS_MESSAGE}"})),
        (422, json.dumps({"message": "No search results quota remaining"})),
        (422, json.dumps([NO_RESULTS_MESSAGE])),
        (422, json.dumps(NO_RESULTS_MESSAGE)),
        (422, '{"message": "' + NO_RESULTS_MESSAGE),
        (422, "<html>Invalid query: " + NO_RESULTS_MESSAGE + "</html>"),
        (401, json.dumps({"message": NO_RESULTS_MESSAGE})),
        (400, json.dumps({"message": NO_RESULTS_MESSAGE})),
        (503, json.dumps({"message": NO_RESULTS_MESSAGE})),
    ],
    indirect=True,
)
async def test_other_http_errors_still_fail_through_block_executor(jina_http_response):
    block = SearchTheWebBlock()

    with pytest.raises(BlockExecutionError, match="Search failed: HTTP") as raised:
        [
            output
            async for output in block.execute(
                {"query": QUERY, "credentials": TEST_CREDENTIALS_INPUT},
                execution_context=ExecutionContext(),
                credentials=TEST_CREDENTIALS,
            )
        ]

    assert isinstance(raised.value.__cause__, (HTTPClientError, HTTPServerError))
    assert block.execution_stats.provider_cost is None
    assert block.execution_stats.provider_cost_type is None
    assert jina_http_response == ["/"]


@pytest.mark.parametrize(
    "jina_http_response, content",
    [((200, "search content"), "search content"), ((200, ""), "")],
    indirect=["jina_http_response"],
)
async def test_success_retains_content_and_cost(jina_http_response, content):
    block = SearchTheWebBlock()
    outputs = [
        output
        async for output in block.execute(
            {"query": QUERY, "credentials": TEST_CREDENTIALS_INPUT},
            execution_context=ExecutionContext(),
            credentials=TEST_CREDENTIALS,
        )
    ]

    assert outputs == [("results", content)]
    assert block.execution_stats.provider_cost == 0.01
    assert block.execution_stats.provider_cost_type == "cost_usd"
    assert jina_http_response == ["/"]


@pytest.mark.parametrize("query", [None, False, 123, [], {}])
@pytest.mark.parametrize("jina_http_response", [(200, "")], indirect=True)
async def test_invalid_query_rejected_before_request(jina_http_response, query):
    block = SearchTheWebBlock()

    with pytest.raises(BlockInputError):
        [
            output
            async for output in block.execute(
                {"query": query, "credentials": TEST_CREDENTIALS_INPUT},
                execution_context=ExecutionContext(),
                credentials=TEST_CREDENTIALS,
            )
        ]

    assert jina_http_response == []
