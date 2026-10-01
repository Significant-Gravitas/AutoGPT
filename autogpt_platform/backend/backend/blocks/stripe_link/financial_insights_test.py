"""Financial-insights block behaviour: the wait for freshly connected accounts,
refused grants, pagination and query building, and tolerant parsing.

Wire contract: Stripe's Link SDK (`_operations.py`, `models.py`) and
https://docs.stripe.com/financial-connections/agents/financial-insights
"""

import logging
from typing import Any
from urllib.parse import parse_qsl

import httpx
import pytest
from pydantic import ValidationError

from backend.blocks.stripe_link import _financial_api, _financial_testdata
from backend.blocks.stripe_link import financial_insights as fi
from backend.blocks.stripe_link._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    LinkAPIError,
    link_api_request,
)

PENDING = {
    "code": "external_data_retrieval_pending",
    "description": "We are still retrieving external financial data.",
}


def txn(n: int, **overrides: Any) -> dict[str, Any]:
    return {
        "id": f"lbctxn_{n}",
        "source_id": "csmrpd_a",
        "amount": -100 * n,
        "currency": "usd",
        "created_date": "2026-06-15",
        "description": f"Merchant {n}",
        "origin": "external_connection",
        "category": None,
        "status": "succeeded",
        **overrides,
    }


def page(items: list[Any], has_more: bool = False) -> dict[str, Any]:
    return {"data": items, "has_more": has_more}


async def run_block(
    block: Any, inputs: dict[str, Any], replies: list[Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run `block` against canned Link replies, recording every request."""
    calls: list[dict[str, Any]] = []
    pending_replies = iter(replies)

    async def fake(credentials, method, path, body=None, params=None):
        calls.append({"method": method, "path": path, "params": list(params or [])})
        reply = next(pending_replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    object.__setattr__(block, "_link_api_request", fake)
    input_data = block.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **inputs}
    )
    outputs = {
        name: value
        async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS)
    }
    return outputs, calls


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """Record the pending-data waits instead of sleeping through them."""
    waited: list[float] = []

    async def fake_sleep(delay: float) -> None:
        waited.append(delay)

    monkeypatch.setattr(_financial_api, "sleep", fake_sleep)
    return waited


# ---------------------------------------------------------------------------
# A freshly connected account (202 external_data_retrieval_pending)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_data_still_loading_is_waited_out_then_returned(waits):
    outputs, calls = await run_block(
        fi.StripeLinkListTransactionsBlock(), {}, [PENDING, PENDING, page([txn(1)])]
    )

    assert "error" not in outputs
    assert [t.id for t in outputs["transactions"]] == ["lbctxn_1"]
    assert len(calls) == 3
    # The same request each time, with the documented back-off between.
    assert calls[0] == calls[1] == calls[2]
    assert waits == list(_financial_api.PENDING_RETRY_DELAYS[:2])


@pytest.mark.asyncio
async def test_data_still_loading_after_the_wait_says_to_try_again_shortly(waits):
    attempts = len(_financial_api.PENDING_RETRY_DELAYS) + 1

    outputs, calls = await run_block(
        fi.StripeLinkGetBalancesBlock(), {}, [PENDING] * attempts
    )

    assert "still loading" in outputs["error"]
    assert "30 seconds" in outputs["error"]
    assert "balances" not in outputs
    # Bounded: no request after the last wait, and the wait stays short
    # enough for a copilot turn.
    assert len(calls) == attempts
    assert sum(waits) <= 30


# ---------------------------------------------------------------------------
# A refused grant: reconnect, don't retry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(
    "block_cls",
    [
        fi.StripeLinkListFinancialAccountsBlock,
        fi.StripeLinkListTransactionsBlock,
        fi.StripeLinkGetBalancesBlock,
    ],
)
async def test_a_refused_grant_asks_the_user_to_reconnect(block_cls, status, waits):
    refused = LinkAPIError(
        f"Link API error ({status})", status_code=status, code="feature_unavailable"
    )

    outputs, calls = await run_block(block_cls(), {}, [refused])

    assert "Reconnect Stripe Link and share" in outputs["error"]
    assert "feature_unavailable" in outputs["error"]
    assert len(calls) == 1 and waits == []


@pytest.mark.asyncio
async def test_other_link_errors_are_reported_as_they_are(waits):
    failure = LinkAPIError("Link API error (500)", status_code=500)

    outputs, calls = await run_block(
        fi.StripeLinkListTransactionsBlock(), {}, [failure]
    )

    assert outputs["error"] == "Link API error (500)"
    assert len(calls) == 1 and waits == []


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transaction_filters_reach_link_under_its_parameter_names():
    _, calls = await run_block(
        fi.StripeLinkListTransactionsBlock(),
        {
            "start_date": "2026-06-01",
            "end_date": " 2026-06-30 ",
            "source_ids": ["csmrpd_a", " csmrpd_b ", "csmrpd_a", ""],
            "origin": "external_connection",
            "category": "groceries",
        },
        [page([])],
    )

    params = calls[0]["params"]
    assert calls[0]["path"] == "/transactions"
    assert ("date_start", "2026-06-01") in params
    assert ("date_end", "2026-06-30") in params
    assert ("origin", "external_connection") in params
    assert ("category", "groceries") in params
    # One repeated `sources[]` per account, trimmed and de-duplicated.
    assert [v for k, v in params if k == "sources[]"] == ["csmrpd_a", "csmrpd_b"]
    assert ("limit", "100") in params


@pytest.mark.asyncio
async def test_unset_filters_are_left_out_rather_than_sent_empty():
    _, calls = await run_block(fi.StripeLinkListTransactionsBlock(), {}, [page([])])

    # `all` is this block's word for "no origin filter"; Link has no such value.
    assert calls[0]["params"] == [("limit", "100")]


@pytest.mark.asyncio
async def test_balances_filter_by_repeated_sources():
    _, calls = await run_block(
        fi.StripeLinkGetBalancesBlock(),
        {"source_ids": ["csmrpd_a", "csmrpd_b"]},
        [page([])],
    )

    assert calls[0]["path"] == "/balances"
    assert [v for k, v in calls[0]["params"] if k == "sources[]"] == [
        "csmrpd_a",
        "csmrpd_b",
    ]


@pytest.mark.asyncio
async def test_repeated_sources_survive_onto_the_wire(monkeypatch):
    """Through the real request helper: a dict of params would keep only the
    last account, and an unencoded `[]` is not what Link parses."""
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=page([txn(1)]))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    block = fi.StripeLinkListTransactionsBlock()
    input_data = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "start_date": "2026-06-01",
            "source_ids": ["csmrpd_a", "csmrpd_b"],
        }
    )

    outputs = {
        name: value
        async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS)
    }

    assert "error" not in outputs
    request = seen[0]
    assert request.url.path == "/transactions"
    assert request.headers["Authorization"] == "Bearer mock-link-access-token"
    query = request.url.query.decode()
    assert "sources%5B%5D=csmrpd_a&sources%5B%5D=csmrpd_b" in query
    assert parse_qsl(query) == [
        ("date_start", "2026-06-01"),
        ("sources[]", "csmrpd_a"),
        ("sources[]", "csmrpd_b"),
        ("limit", "100"),
    ]


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_limit_beyond_one_page_follows_the_cursor_with_the_same_filters():
    first = [txn(n) for n in range(1, 101)]
    second = [txn(n) for n in range(101, 151)]

    outputs, calls = await run_block(
        fi.StripeLinkListTransactionsBlock(),
        {"limit": 150, "start_date": "2026-06-01"},
        [page(first, has_more=True), page(second, has_more=True)],
    )

    assert len(outputs["transactions"]) == 150
    assert outputs["has_more"] is True
    assert outputs["next_cursor"] == "lbctxn_150"
    assert calls[0]["params"] == [("date_start", "2026-06-01"), ("limit", "100")]
    assert calls[1]["params"] == [
        ("date_start", "2026-06-01"),
        ("limit", "50"),
        ("starting_after", "lbctxn_100"),
    ]


@pytest.mark.asyncio
async def test_the_last_page_ends_reading_with_no_cursor():
    outputs, calls = await run_block(
        fi.StripeLinkListTransactionsBlock(),
        {"limit": 500},
        [page([txn(1), txn(2)], has_more=False)],
    )

    assert len(calls) == 1
    assert outputs["has_more"] is False
    assert outputs["next_cursor"] == ""


@pytest.mark.asyncio
async def test_a_cursor_resumes_where_the_previous_run_stopped():
    _, calls = await run_block(
        fi.StripeLinkListTransactionsBlock(),
        {"starting_after": " lbctxn_150 "},
        [page([])],
    )

    assert ("starting_after", "lbctxn_150") in calls[0]["params"]


@pytest.mark.asyncio
async def test_more_than_was_asked_for_is_cut_and_reported_as_more():
    outputs, _ = await run_block(
        fi.StripeLinkListTransactionsBlock(),
        {"limit": 2},
        [page([txn(1), txn(2), txn(3)], has_more=False)],
    )

    assert [t.id for t in outputs["transactions"]] == ["lbctxn_1", "lbctxn_2"]
    assert outputs["has_more"] is True
    assert outputs["next_cursor"] == "lbctxn_2"


@pytest.mark.asyncio
async def test_more_reported_with_an_empty_page_stops_instead_of_looping():
    outputs, calls = await run_block(
        fi.StripeLinkListTransactionsBlock(),
        {"limit": 500},
        [page([], has_more=True)],
    )

    assert len(calls) == 1
    assert outputs["transactions"] == []
    assert outputs["next_cursor"] == ""


@pytest.mark.asyncio
async def test_accounts_are_read_across_pages_by_their_ids():
    outputs, calls = await run_block(
        fi.StripeLinkListFinancialAccountsBlock(),
        {},
        [
            page([{"id": "csmrpd_a"}], has_more=True),
            page([{"id": "csmrpd_b"}], has_more=False),
        ],
    )

    assert [a.id for a in outputs["accounts"]] == ["csmrpd_a", "csmrpd_b"]
    assert ("starting_after", "csmrpd_a") in calls[1]["params"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_statuses_and_explicit_nulls_pass_through():
    """`status` is an open set: new values must not fail the read or be
    rewritten. Nulls stay null where they mean something, and become ""
    where a string output is expected."""
    outputs, _ = await run_block(
        fi.StripeLinkListTransactionsBlock(),
        {},
        [
            page(
                [
                    txn(1, status="succeeded"),
                    txn(2, status="requires_review_v2"),
                    txn(3, status=None, source_id=None, description=None),
                    txn(4, origin="some_future_origin", category="dining"),
                ]
            )
        ],
    )

    transactions = outputs["transactions"]
    assert [t.status for t in transactions] == [
        "succeeded",
        "requires_review_v2",
        "",
        "succeeded",
    ]
    assert transactions[2].source_id is None
    assert transactions[2].description == ""
    assert transactions[3].origin == "some_future_origin"
    assert transactions[3].category == "dining"


@pytest.mark.asyncio
async def test_a_bare_list_of_transactions_is_read_as_the_final_page():
    outputs, _ = await run_block(
        fi.StripeLinkListTransactionsBlock(), {}, [[txn(1), txn(2)]]
    )

    assert [t.id for t in outputs["transactions"]] == ["lbctxn_1", "lbctxn_2"]
    assert outputs["has_more"] is False


@pytest.mark.asyncio
async def test_a_malformed_transaction_fails_without_repeating_its_contents():
    """A total that silently skipped a record would be wrong, so the read
    fails; the message names the field but not the person's data."""
    broken = txn(1, description="PRIVATE CLINIC 4417")
    del broken["amount"]

    outputs, _ = await run_block(
        fi.StripeLinkListTransactionsBlock(), {}, [page([broken])]
    )

    assert "transaction" in outputs["error"]
    assert "amount" in outputs["error"]
    assert "PRIVATE CLINIC" not in outputs["error"]
    assert "lbctxn_1" not in outputs["error"]


@pytest.mark.asyncio
async def test_a_page_without_data_is_an_error_not_an_empty_history():
    outputs, _ = await run_block(
        fi.StripeLinkListTransactionsBlock(), {}, [{"object": "list"}]
    )

    assert "data" in outputs["error"]
    assert "transactions" not in outputs


@pytest.mark.asyncio
async def test_balances_flatten_cash_and_credit_amounts():
    outputs, _ = await run_block(
        fi.StripeLinkGetBalancesBlock(),
        {},
        [
            page(
                [
                    {
                        "source_id": "csmrpd_a",
                        "type": "cash",
                        "current": 31005,
                        "currency": "usd",
                        "as_of": "2026-07-15T00:00:00Z",
                        "cash": {"available": {"usd": 30005}},
                    },
                    {
                        "source_id": "csmrpd_b",
                        "type": "credit",
                        "current": 12000,
                        "currency": "usd",
                        "as_of": "2026-07-15T00:00:00Z",
                        "credit": {"used": {"usd": 12550}},
                    },
                    {
                        "source_id": "csmrpd_c",
                        "type": "credit",
                        "current": 0,
                        "currency": "usd",
                        "as_of": None,
                        "credit": {"used": None},
                    },
                ]
            )
        ],
    )

    cash, credit, empty_credit = outputs["balances"]
    assert (cash.available, cash.used) == ({"usd": 30005}, None)
    assert (credit.available, credit.used) == (None, {"usd": 12550})
    assert (empty_credit.used, empty_credit.as_of) == (None, "")


@pytest.mark.asyncio
async def test_accounts_keep_only_known_fields():
    """Projected like payment methods: whatever Link adds to a source later
    must not land in a persisted output by default."""
    outputs, _ = await run_block(
        fi.StripeLinkListFinancialAccountsBlock(),
        {},
        [
            page(
                [
                    {
                        "id": "csmrpd_a",
                        "name": "BANK CHECKING",
                        "type": "bank_account",
                        "capabilities": {
                            "balances": {"status": "eligible"},
                            "transactions": {"status": "pending"},
                            "future_capability": None,
                        },
                        "external_connection": {"status": "active"},
                        "bank_account": {
                            "last4": "6789",
                            "bank_name": "Example Bank",
                            "account_number": "000123456789",
                            "routing_number": "110000000",
                        },
                        "granted_actions": ["read_balances", None],
                        "some_future_field": "x",
                    },
                    {
                        "id": "csmrpd_b",
                        "type": "card",
                        "card": {"brand": "visa", "last4": "4242", "number": "4242"},
                        "capabilities": None,
                        "external_connection": None,
                    },
                ]
            )
        ],
    )

    bank, card = outputs["accounts"]
    assert bank.model_dump() == {
        "id": "csmrpd_a",
        "name": "BANK CHECKING",
        "type": "bank_account",
        "last4": "6789",
        "brand": "",
        "bank_name": "Example Bank",
        "capabilities": {"balances": "eligible", "transactions": "pending"},
        "connection_status": "active",
        "granted_actions": ["read_balances"],
    }
    assert "000123456789" not in str(outputs)
    assert (card.brand, card.last4, card.capabilities) == ("visa", "4242", {})


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["2026/06/01", "20260601", "2026-02-30", "June 1"])
def test_dates_must_be_real_dates_written_yyyy_mm_dd(bad):
    with pytest.raises(ValidationError, match="start_date"):
        fi.StripeLinkListTransactionsBlock.Input.model_validate(
            {"credentials": TEST_CREDENTIALS_INPUT, "start_date": bad}
        )


def test_a_reversed_date_range_is_rejected():
    """Link would answer it with an empty list, which reads as no spending."""
    with pytest.raises(ValidationError, match="after end_date"):
        fi.StripeLinkListTransactionsBlock.Input.model_validate(
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "start_date": "2026-07-01",
                "end_date": "2026-06-01",
            }
        )


@pytest.mark.parametrize("limit", [0, fi.MAX_TRANSACTIONS + 1])
def test_the_transaction_limit_is_bounded(limit):
    with pytest.raises(ValidationError, match="limit"):
        fi.StripeLinkListTransactionsBlock.Input.model_validate(
            {"credentials": TEST_CREDENTIALS_INPUT, "limit": limit}
        )


# ---------------------------------------------------------------------------
# Privacy and the request helper
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_nothing_from_a_response_body_is_logged(waits, caplog):
    caplog.set_level(logging.DEBUG)

    await run_block(
        fi.StripeLinkListTransactionsBlock(),
        {},
        [PENDING, page([txn(1, description="PRIVATE CLINIC 4417", amount=-31337)])],
    )

    assert "PRIVATE CLINIC" not in caplog.text
    assert "31337" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, code",
    [
        (
            {"error": {"message": "Access revoked", "code": "feature_unavailable"}},
            "feature_unavailable",
        ),
        (
            {"code": "feature_unavailable", "description": "Not shared"},
            "feature_unavailable",
        ),
        ({"error": "just a string"}, ""),
    ],
)
async def test_link_errors_keep_their_status_and_code(monkeypatch, body, code):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            **kwargs,
            transport=httpx.MockTransport(lambda _: httpx.Response(403, json=body)),
        ),
    )

    with pytest.raises(LinkAPIError) as raised:
        await link_api_request(TEST_CREDENTIALS, "GET", "/sources")

    assert raised.value.status_code == 403
    assert raised.value.code == code
    # Still a RuntimeError with the familiar message, for existing callers.
    assert isinstance(raised.value, RuntimeError)
    assert str(raised.value).startswith("Link API error (403)")


def test_built_in_test_data_round_trips_through_the_parsers():
    """The built-in tests feed wire-shaped data through the real parsers; this
    keeps the expected models honest if either side changes."""
    data = _financial_testdata
    assert _financial_api.parse_account(data.TEST_SOURCE_WIRE) == data.TEST_ACCOUNT
    assert _financial_api.parse_balance(data.TEST_BALANCE_WIRE) == data.TEST_BALANCE
    assert (
        _financial_api.parse_transaction(data.TEST_TRANSACTION.model_dump())
        == data.TEST_TRANSACTION
    )
