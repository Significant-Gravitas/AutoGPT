"""
Stripe Link — reading financial-insights data for the blocks in
``financial_insights.py``.

The data comes from Link's API (api.link.com), not the Stripe API, and the wire
contract follows Stripe's Link SDK: ``GET /sources``, ``/transactions`` and
``/balances`` each return a ``{data, has_more}`` page with cursor pagination.
This module owns the parts every one of those reads shares:

- following ``has_more`` with the same filters, bounded so a cursor that never
  advances cannot loop;
- the wait for a freshly connected account, which Link signals with a 202
  ``external_data_retrieval_pending`` for up to about 30 seconds;
- turning a refused grant into an instruction to reconnect, since no retry
  can fix it.

Nothing here logs a response body: these are people's balances and
transactions.
"""

import logging
import math
import re
from asyncio import sleep
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from backend.blocks.stripe_link._auth import LinkAPIError, StripeLinkCredentials
from backend.blocks.stripe_link._financial_models import (
    LinkBalance,
    LinkFinancialAccount,
    LinkTransaction,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
LinkRequest = Callable[..., Awaitable[Any]]

PENDING_CODE = "external_data_retrieval_pending"
# Seconds between attempts while Link is still pulling a freshly connected
# account's data. Stripe says it is usually ready within 30 seconds; about 14
# seconds of waiting covers the common case without holding a run, or a
# copilot turn, for long. Past it the user is asked to try again.
PENDING_RETRY_DELAYS: tuple[float, ...] = (2.0, 4.0, 8.0)
# Link serves at most this many records per request.
MAX_PAGE_SIZE = 100
# Accounts and their balances are read whole: a consumer has a handful, and the
# cap only bounds a misbehaving cursor.
MAX_ACCOUNTS = 500
# 401 after the credentials manager has refreshed on acquire means the grant
# itself is gone; 403 (`feature_unavailable`) means it does not cover this data.
ACCESS_REFUSED_STATUSES = frozenset({401, 403})

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class LinkAccessError(RuntimeError):
    """The grant does not cover this data: revoked, or the account was never
    shared. Retrying cannot help; the user has to reconnect."""


class LinkDataPendingError(RuntimeError):
    """Link was still retrieving a freshly connected account's data when the
    wait ran out."""


async def read_pages(
    api_request: LinkRequest,
    credentials: StripeLinkCredentials,
    path: str,
    filters: Sequence[tuple[str, str]],
    *,
    parse: Callable[[Any], T],
    cursor_of: Callable[[T], str],
    max_items: int,
    starting_after: str = "",
) -> tuple[list[T], bool]:
    """Read up to `max_items` records, returning them and whether Link holds
    more beyond them.

    Every request repeats the same filters and moves only `starting_after`, as
    Link's cursors require. The number of requests is bounded by `max_items`,
    so a cursor that never advances cannot keep the block looping.
    """
    items: list[T] = []
    cursor = starting_after
    has_more = False
    for _ in range(math.ceil(max_items / MAX_PAGE_SIZE)):
        query = [*filters, ("limit", str(min(MAX_PAGE_SIZE, max_items - len(items))))]
        if cursor:
            query.append(("starting_after", cursor))
        payload = await get_page(api_request, credentials, path, query)
        batch, has_more = _read_page(payload, parse)
        items += batch
        cursor = cursor_of(batch[-1]) if batch else ""
        if not has_more or not cursor or len(items) >= max_items:
            break
    if len(items) > max_items:
        # Link sent more than was asked for. Cut there and say so, so the
        # caller resumes from the last record it actually received.
        del items[max_items:]
        has_more = True
    return items, has_more


async def read_all(
    api_request: LinkRequest,
    credentials: StripeLinkCredentials,
    path: str,
    filters: Sequence[tuple[str, str]],
    *,
    parse: Callable[[Any], T],
    cursor_of: Callable[[T], str],
) -> list[T]:
    """Read every record of a short list (accounts, balances)."""
    items, has_more = await read_pages(
        api_request,
        credentials,
        path,
        filters,
        parse=parse,
        cursor_of=cursor_of,
        max_items=MAX_ACCOUNTS,
    )
    if has_more:
        logger.warning(f"Link {path} listed more than {MAX_ACCOUNTS} records")
    return items


async def get_page(
    api_request: LinkRequest,
    credentials: StripeLinkCredentials,
    path: str,
    params: Sequence[tuple[str, str]],
) -> Any:
    """GET one page, waiting out a freshly connected account's data load."""
    for delay in (*PENDING_RETRY_DELAYS, None):
        try:
            payload = await api_request(credentials, "GET", path, params=list(params))
        except LinkAPIError as e:
            if e.status_code in ACCESS_REFUSED_STATUSES:
                raise LinkAccessError(_access_refused_message(e)) from e
            raise
        if not _is_pending(payload):
            return payload
        if delay is None:
            break
        logger.debug("Link %s data still loading; retrying in %ss", path, delay)
        await sleep(delay)
    raise LinkDataPendingError(
        "Link is still loading data for a newly connected account. Try again "
        "in about 30 seconds."
    )


def build_filters(
    *,
    start_date: str = "",
    end_date: str = "",
    origin: str = "",
    category: str = "",
    source_ids: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Link's filter parameters, with unset ones left out.

    The date range goes under the names Link reads (`date_start`/`date_end`,
    which the SDK maps its start/end dates to), and each account is its own
    repeated `sources[]` pair.
    """
    named = {
        "date_start": start_date,
        "date_end": end_date,
        "origin": origin,
        "category": category,
    }
    return [(key, value) for key, value in named.items() if value] + [
        ("sources[]", source_id) for source_id in source_ids
    ]


def iso_date(value: str) -> str:
    """'' or a real calendar date written YYYY-MM-DD.

    `date.fromisoformat` alone also accepts forms like 20260601 and week dates,
    which Link would not read as a date.
    """
    value = value.strip()
    if not value:
        return ""
    if not _ISO_DATE.fullmatch(value):
        raise ValueError(f"expected a date written YYYY-MM-DD, got {value!r}")
    date.fromisoformat(value)  # rejects impossible dates such as 2026-02-30
    return value


def unique_ids(values: Sequence[str]) -> list[str]:
    """Trimmed, non-empty and de-duplicated, in the order given."""
    return list(dict.fromkeys(v.strip() for v in values if v.strip()))


def parse_transaction(item: Any) -> LinkTransaction:
    return _validate(LinkTransaction, item, "transaction")


def parse_balance(item: Any) -> LinkBalance:
    record = _record(item, "balance")
    return _validate(
        LinkBalance,
        {
            **record,
            "available": _as_dict(record.get("cash")).get("available"),
            "used": _as_dict(record.get("credit")).get("used"),
        },
        "balance",
    )


def parse_account(item: Any) -> LinkFinancialAccount:
    record = _record(item, "account")
    # A source carries `card` or `bank_account`, never both. Projected to
    # known fields so whatever Link adds to them later stays out of persisted
    # outputs.
    details = {**_as_dict(record.get("card")), **_as_dict(record.get("bank_account"))}
    actions = record.get("granted_actions")
    return _validate(
        LinkFinancialAccount,
        {
            "id": record.get("id"),
            "name": record.get("name"),
            "type": record.get("type"),
            "last4": details.get("last4"),
            "brand": details.get("brand"),
            "bank_name": details.get("bank_name"),
            "capabilities": {
                name: capability.get("status")
                for name, capability in _as_dict(record.get("capabilities")).items()
                if isinstance(capability, dict)
                and isinstance(capability.get("status"), str)
            },
            "connection_status": _as_dict(record.get("external_connection")).get(
                "status"
            ),
            "granted_actions": [
                action
                for action in (actions if isinstance(actions, list) else [])
                if isinstance(action, str)
            ],
        },
        "account",
    )


def _read_page(payload: Any, parse: Callable[[Any], T]) -> tuple[list[T], bool]:
    """Parse a `{data, has_more}` page. The transactions endpoint may also
    answer with a bare list, which the SDK reads as a final page."""
    if isinstance(payload, list):
        return [parse(item) for item in payload], False
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("Link returned a page without a `data` list")
    return [parse(item) for item in data], payload.get("has_more") is True


def _validate(model: type[T], data: Any, what: str) -> T:
    """Validate a record, reporting only where it failed.

    Field locations and error kinds, never values: the input is someone's
    financial data, and this message becomes a persisted block output.
    """
    try:
        return model.model_validate(data)
    except ValidationError as e:
        problems = ", ".join(
            f"{'.'.join(str(part) for part in error['loc']) or what}: {error['type']}"
            for error in e.errors(include_input=False, include_url=False)[:3]
        )
        raise ValueError(
            f"Link returned a {what} this block cannot read ({problems})"
        ) from None


def _is_pending(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("code") == PENDING_CODE


def _record(item: Any, what: str) -> dict[str, Any]:
    """A top-level record must be an object; anything else would otherwise
    parse into an empty record that reads as real data."""
    if not isinstance(item, dict):
        raise ValueError(f"Link returned a {what} this block cannot read")
    return item


def _as_dict(value: Any) -> dict[str, Any]:
    """`value` if it is an object, else {}: Link sends explicit nulls, and a
    `None` part-way down a chain would otherwise raise AttributeError."""
    return value if isinstance(value, dict) else {}


def _access_refused_message(error: LinkAPIError) -> str:
    code = f" ({error.code})" if error.code else ""
    return (
        f"Stripe Link refused access to this financial data: {error}{code}. "
        "Reconnect Stripe Link and share the bank and card accounts the agent "
        "should read; retrying will not help until then."
    )
