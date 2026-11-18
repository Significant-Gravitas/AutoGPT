"""
Built-in test data for the financial-insights blocks, in Link's wire shape and
as the blocks emit it. The block tests feed the wire shapes through the real
parsers, so each pair has to agree.
"""

from typing import Any

from backend.blocks.stripe_link._financial_models import (
    LinkBalance,
    LinkFinancialAccount,
    LinkTransaction,
)

TEST_SOURCE_WIRE: dict[str, Any] = {
    "id": "csmrpd_test",
    "name": "BANK SAVINGS",
    "type": "bank_account",
    "capabilities": {
        "balances": {"status": "eligible"},
        "transactions": {"status": "eligible"},
    },
    "external_connection": {"status": "active"},
    "bank_account": {"last4": "5115"},
    "granted_actions": ["read_balances", "read_external_transactions"],
}
TEST_ACCOUNT = LinkFinancialAccount(
    id="csmrpd_test",
    name="BANK SAVINGS",
    type="bank_account",
    last4="5115",
    capabilities={"balances": "eligible", "transactions": "eligible"},
    connection_status="active",
    granted_actions=["read_balances", "read_external_transactions"],
)

TEST_TRANSACTION = LinkTransaction(
    id="lbctxn_test",
    source_id="csmrpd_test",
    amount=-4999,
    currency="usd",
    created_date="2026-06-15",
    description="ACME Coffee Shop",
    origin="external_connection",
    category=None,
    status="succeeded",
)

TEST_BALANCE_WIRE: dict[str, Any] = {
    "source_id": "csmrpd_test",
    "type": "cash",
    "current": 31005,
    "currency": "usd",
    "as_of": "2026-07-15T00:00:00Z",
    "cash": {"available": {"usd": 30005}},
}
TEST_BALANCE = LinkBalance(
    source_id="csmrpd_test",
    type="cash",
    current=31005,
    currency="usd",
    available={"usd": 30005},
    as_of="2026-07-15T00:00:00Z",
)
