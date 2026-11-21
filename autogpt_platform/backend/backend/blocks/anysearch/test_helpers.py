"""Shared fixtures and builders for the AnySearch block tests."""

import inspect

from pydantic import SecretStr

from backend.blocks.anysearch.domains import AnySearchDomainsBlock
from backend.blocks.anysearch.extract import AnySearchExtractBlock
from backend.blocks.anysearch.parallel_search import AnySearchParallelSearchBlock
from backend.blocks.anysearch.search import AnySearchBlock
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="anysearch",
    api_key=SecretStr("mock-anysearch-api-key"),
    title="Mock Anysearch API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}

# (block class, minimal valid input, mockable request method) per block that
# honours the auth discriminator - the guard tests in test_auth.py parametrize
# over this table.
_AUTH_CASES = [
    (AnySearchBlock, {"query": "q"}, "_search"),
    (AnySearchParallelSearchBlock, {"queries": ["q"]}, "_search"),
    (AnySearchExtractBlock, {"url": "https://x.test"}, "_extract"),
    (AnySearchDomainsBlock, {}, "_catalog"),
]


def _mock_block(block, mocks: dict):
    """Apply mocks to a block's methods, wrapping sync mocks as async."""
    for name, mock_fn in mocks.items():
        original = getattr(block, name)
        if inspect.iscoroutinefunction(original):

            async def async_mock(*args, _fn=mock_fn, **kwargs):
                return _fn(*args, **kwargs)

            setattr(block, name, async_mock)
        else:  # pragma: no cover - all mocked seams are coroutine fns
            setattr(block, name, mock_fn)


def _raise(exc: Exception):
    """Return a callable which raises the given exception."""

    def _raiser(*args, **kwargs):
        raise exc

    return _raiser


async def _collect(
    block, input_data: dict, credentials: APIKeyCredentials | None = TEST_CREDENTIALS
) -> dict:
    """Run ``block.execute`` and gather its yielded outputs into a dict."""
    outputs = {}
    async for name, value in block.execute(
        input_data,
        credentials=credentials,
        execution_context=ExecutionContext(),
    ):
        outputs[name] = value
    return outputs


def _row(title: str = "AutoGPT", url: str = "https://agpt.co") -> dict:
    return {
        "title": title,
        "url": url,
        "snippet": "AutoGPT is a platform for building AI agents.",
        "content": None,
    }


def _search_response(results, request_id: str = "req-1") -> dict:
    return {
        "code": 0,
        "message": "success",
        "request_id": request_id,
        "data": {"results": results},
    }


def _catalog_response(domains, request_id: str = "req-1") -> dict:
    return {
        "code": 0,
        "message": "success",
        "request_id": request_id,
        "data": {"domains": domains, "topology_version": 9},
    }


def _spy(captured: list):
    """Credential-capture spy returning a response shaped for the method:
    str -> extract, list/None -> catalog, dict payload -> search."""

    def spy(creds, arg):
        captured.append(creds)
        if isinstance(arg, str):
            data = {"url": arg, "title": "t", "content": "c"}
            return {"code": 0, "message": "success", "request_id": "r", "data": data}
        if not isinstance(arg, dict):
            return _catalog_response([])
        return _search_response([])

    return spy
