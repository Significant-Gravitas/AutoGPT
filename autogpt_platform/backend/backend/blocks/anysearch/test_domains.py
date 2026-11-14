"""Unit tests for AnySearchDomainsBlock (live catalog discovery)."""

import pytest

from backend.blocks.anysearch.domains import AnySearchDomainsBlock
from backend.blocks.anysearch.test_helpers import (
    TEST_CREDENTIALS_INPUT,
    _catalog_response,
    _collect,
    _mock_block,
    _raise,
)
from backend.util.exceptions import BlockExecutionError
from backend.util.request import HTTPClientError, HTTPServerError

_CATALOG = [
    {"domain": "academic", "description": "Scholarly search", "sub_domain_count": 5},
    {"domain": "code", "description": "Developer docs", "sub_domain_count": 2},
]


@pytest.mark.asyncio
async def test_domains_lists_full_catalog():
    """No ``domains`` input -> GET /v1/domains, entries passed through
    verbatim with topology_version and request_id."""
    captured = {}

    def spy(creds, domains):
        captured["domains"] = domains
        return _catalog_response(_CATALOG, request_id="req-cat")

    block = AnySearchDomainsBlock()
    _mock_block(block, {"_catalog": spy})

    outputs = await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT})

    assert captured["domains"] is None
    assert outputs["domains"] == _CATALOG
    assert outputs["topology_version"] == 9
    assert outputs["request_id"] == "req-cat"


@pytest.mark.asyncio
async def test_domains_selected_domains_hit_sub_domains():
    """Named domains -> GET /v1/sub-domains with the enum values as
    repeated ``domain`` params (client-side, verified in test_api)."""
    captured = {}

    detail = [
        {
            "domain": "code",
            "sub_domains": [
                {"sub_domain": "code.doc", "params": {"library": {"required": True}}}
            ],
        }
    ]

    def spy(creds, domains):
        captured["domains"] = domains
        return _catalog_response(detail)

    block = AnySearchDomainsBlock()
    _mock_block(block, {"_catalog": spy})

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "domains": ["code", "finance"]},
    )

    assert captured["domains"] == ["code", "finance"]
    assert outputs["domains"] == detail


@pytest.mark.asyncio
async def test_domains_empty_list_lists_catalog():
    """An empty selection means the full catalog - the endpoint requires
    a domain, so no /v1/sub-domains call may be made."""
    captured = {}

    def spy(creds, domains):
        captured["domains"] = domains
        return _catalog_response(_CATALOG)

    block = AnySearchDomainsBlock()
    _mock_block(block, {"_catalog": spy})

    outputs = await _collect(
        block, {"credentials": TEST_CREDENTIALS_INPUT, "domains": []}
    )

    assert captured["domains"] is None
    assert outputs["domains"] == _CATALOG


@pytest.mark.asyncio
async def test_domains_api_error_raises_block_error():
    block = AnySearchDomainsBlock()
    _mock_block(block, {"_catalog": _raise(RuntimeError("catalog boom"))})

    with pytest.raises(BlockExecutionError, match="Domains discovery failed"):
        await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT})


@pytest.mark.asyncio
async def test_domains_runs_anonymously_without_credentials():
    captured = {}

    def spy(creds, domains):
        captured["creds"] = creds
        return _catalog_response(_CATALOG)

    block = AnySearchDomainsBlock()
    _mock_block(block, {"_catalog": spy})
    outputs = await _collect(block, {"auth": "anonymous"}, credentials=None)
    assert captured["creds"] is None
    assert outputs["domains"] == _CATALOG


@pytest.mark.asyncio
async def test_domains_missing_key_raises_block_error():
    """code==0 without data.domains is a response error, not an empty
    catalog."""
    block = AnySearchDomainsBlock()
    _mock_block(
        block,
        {
            "_catalog": lambda *a, **k: {
                "code": 0,
                "message": "success",
                "request_id": "r",
                "data": {"topology_version": 9},
            }
        },
    )
    with pytest.raises(BlockExecutionError, match="missing data.domains"):
        await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [None, "not-a-list", [{"domain": "code"}, "oops"]],
    ids=["null", "str", "non_dict_entry"],
)
async def test_domains_malformed_domains_raise_block_error(bad):
    """data.domains must be a list whose elements are entry dicts."""

    def fn(creds, domains):
        return {
            "code": 0,
            "message": "success",
            "request_id": "r",
            "data": {"domains": bad, "topology_version": 9},
        }

    block = AnySearchDomainsBlock()
    _mock_block(block, {"_catalog": fn})

    with pytest.raises(BlockExecutionError, match="malformed AnySearch catalog"):
        await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT})


@pytest.mark.asyncio
async def test_domains_server_error_carries_stale_catalog_hint():
    """A 5xx from the catalog endpoint surfaces the stale-selection hint:
    a domain our enum still lists can be retired upstream, and it fails
    the whole call rather than returning partial results."""
    block = AnySearchDomainsBlock()
    _mock_block(
        block,
        {
            "_catalog": _raise(
                HTTPServerError("HTTP 502 Error: Bad Gateway, Body: upstream", 502)
            )
        },
    )

    with pytest.raises(BlockExecutionError, match="upstream catalog error"):
        await _collect(
            block,
            {"credentials": TEST_CREDENTIALS_INPUT, "domains": ["code"]},
        )

    _mock_block(
        block,
        {
            "_catalog": _raise(
                HTTPClientError("HTTP 400 Error: Bad Request, Body: bad", 400)
            )
        },
    )
    with pytest.raises(BlockExecutionError) as exc:
        await _collect(
            block,
            {"credentials": TEST_CREDENTIALS_INPUT, "domains": ["code"]},
        )
    assert "upstream catalog error" not in str(exc.value)
