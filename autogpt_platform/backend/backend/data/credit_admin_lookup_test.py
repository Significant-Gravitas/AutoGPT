"""Emails are looked up by user ID in chunks, a deleted account has none, and
a search matches user ID, email or name literally, newest first."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import CreditTransactionType

from backend.data import credit_admin_lookup


@pytest.mark.asyncio
async def test_emails_come_back_only_for_accounts_that_exist(monkeypatch):
    users = MagicMock()
    users.prisma.return_value.find_many = AsyncMock(
        return_value=[SimpleNamespace(id="u1", email="u1@example.com")]
    )
    monkeypatch.setattr(credit_admin_lookup, "User", users)

    emails = await credit_admin_lookup.user_emails(["u1", "gone", "u1"])

    assert emails == {"u1": "u1@example.com"}
    where = users.prisma.return_value.find_many.await_args.kwargs["where"]
    assert where == {"id": {"in": ["u1", "gone"]}}


@pytest.mark.asyncio
async def test_many_ids_are_looked_up_in_chunks(monkeypatch):
    users = MagicMock()
    users.prisma.return_value.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr(credit_admin_lookup, "User", users)

    await credit_admin_lookup.user_emails(f"u{i}" for i in range(2500))

    assert users.prisma.return_value.find_many.await_count == 3


@pytest.mark.asyncio
async def test_a_search_matches_literally_and_pages(monkeypatch):
    query = AsyncMock(side_effect=[[{"key": "tx-1", "user_id": "u1"}], [{"n": 31}]])
    monkeypatch.setattr(credit_admin_lookup, "query_raw_with_schema", query)

    keys, total = await credit_admin_lookup.search_history_keys(
        "50%_off",
        include_inactive=False,
        transaction_filter=CreditTransactionType.TOP_UP,
        skip=20,
        take=10,
    )

    assert (keys, total) == ([("tx-1", "u1")], 31)
    page_sql, *page_args = query.await_args_list[0].args
    assert page_args == ["%50\\%\\_off%", False, "TOP_UP", 20, 10]
    assert 'u."email" ILIKE $1' in page_sql
    assert 'ORDER BY ct."createdAt" DESC' in page_sql
    assert query.await_args_list[1].args[1:] == tuple(page_args[:3])
