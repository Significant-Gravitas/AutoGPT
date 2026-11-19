"""The ledger keeps IDs and amounts and loses everything personal: the block
inputs, free-text reasons and the customer details of stored Stripe objects.
What the scrub does to real rows is checked in `account_erasure_db_test.py`."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_erasure import ledger

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"


@pytest.mark.parametrize(
    "key",
    ["input", "reason", "customer_details", "customer_email", "billing_details"],
)
def test_personal_keys_are_not_kept(key):
    assert key not in ledger.LEDGER_KEYS


@pytest.mark.parametrize(
    "key",
    ["payment_intent", "stripe_invoice_id", "amount", "currency", "block_id"],
)
def test_ids_and_amounts_are_kept(key):
    assert key in ledger.LEDGER_KEYS


def test_the_statement_survives_the_schema_prefix_formatting():
    sql = ledger._SCRUB.format(schema_prefix='"platform".', schema="platform")
    assert sql.lstrip().startswith('UPDATE "platform"."CreditTransaction" ct')
    # Only rows that would change, so a rerun is a no-op.
    assert "IS DISTINCT FROM" in sql


@pytest.mark.asyncio
async def test_the_scrub_is_scoped_to_the_account(monkeypatch):
    execute = AsyncMock(return_value=2)
    monkeypatch.setattr(ledger, "execute_raw_with_schema", execute)
    tx = MagicMock()

    assert await ledger.scrub_ledger(USER_ID, client=tx) == 2

    sql, user_id, keys = execute.await_args.args
    assert 'ct."userId" = $1' in sql
    assert (user_id, keys) == (USER_ID, ledger.LEDGER_KEYS)
    assert execute.await_args.kwargs["client"] is tx
