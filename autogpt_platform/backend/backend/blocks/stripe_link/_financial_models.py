"""
Stripe Link — the records the financial-insights blocks emit.

Field names follow Link's wire format, flattened where Link wraps a single
value: a source's `capabilities.balances.status` becomes
`capabilities["balances"]`, `external_connection.status` becomes
`connection_status`, and a balance's `cash.available` and `credit.used` become
`available` and `used`. Parsing from the wire lives in `_financial_api.py`.

The field descriptions are what an agent reads in the output schema, so they
carry the units and sign conventions it needs to answer correctly.
"""

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _null_as_empty(value: Any) -> Any:
    return "" if value is None else value


# Link sends explicit nulls for text it does not have; "" keeps each field a
# plain string for graphs and for the output schema check.
Text = Annotated[str, BeforeValidator(_null_as_empty)]


class _LinkRecord(BaseModel):
    # Validation errors never repeat their input: it is someone's financial
    # data, and error text ends up in persisted outputs and logs.
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)


class LinkFinancialAccount(_LinkRecord):
    """A bank account or card the user shared through Link (a Link "source")."""

    id: Text = Field(
        default="",
        description="Account ID (csmrpd_...). Pass it in `source_ids` to List "
        "Transactions or Get Balances to read only this account.",
    )
    name: Text = Field(default="", description="Display name, e.g. BANK SAVINGS")
    type: Text = Field(default="", description="`bank_account` or `card`")
    last4: Text = Field(
        default="", description="Last four digits of the account or card number"
    )
    brand: Text = Field(
        default="", description="Card brand, for a card, when Link provides it"
    )
    bank_name: Text = Field(
        default="",
        description="Name of the bank, for a bank account, when Link provides it",
    )
    capabilities: dict[str, str] = Field(
        default_factory=dict,
        description='Readiness of each kind of data, e.g. {"balances": '
        '"eligible", "transactions": "pending"}. Only `eligible` data can be '
        "read yet; `pending` means Link is still loading it.",
    )
    connection_status: Text = Field(
        default="",
        description="State of the connection to the bank or card issuer, "
        "e.g. `active`",
    )
    granted_actions: list[str] = Field(
        default_factory=list,
        description="What the user allowed for this account: read_balances, "
        "read_external_transactions, read_link_transactions and/or "
        "read_source_details",
    )


class LinkTransaction(_LinkRecord):
    """One transaction from a shared account, or a purchase made through Link."""

    id: str = Field(description="Transaction ID (lbctxn_...)")
    source_id: str | None = Field(
        default=None,
        description="ID of the account it belongs to (see List Financial "
        "Accounts). Null when Link does not tie it to an account; do not "
        "guess one from the description.",
    )
    amount: int = Field(
        description="Amount in the currency's smallest unit, e.g. cents for "
        "USD: -4999 is $49.99 spent. Negative is money leaving the account, "
        "positive is money coming in."
    )
    currency: str = Field(
        description="Three-letter ISO currency code in lowercase, e.g. usd"
    )
    created_date: str = Field(description="Date of the transaction, YYYY-MM-DD")
    description: Text = Field(
        default="", description="Merchant or statement description"
    )
    origin: Text = Field(
        default="",
        description="`external_connection` for a transaction from a connected "
        "bank or card, `link` for a purchase made through Link",
    )
    category: str | None = Field(
        default=None,
        description="Spending category when Link has one; often null",
    )
    status: Text = Field(
        default="",
        description="Settlement status as Link reports it, e.g. `succeeded`. "
        "New values can appear without notice: treat an unfamiliar one as "
        "information, not as a failure.",
    )


class LinkBalance(_LinkRecord):
    """The current balance of one shared account."""

    source_id: str = Field(
        description="ID of the account (see List Financial Accounts)"
    )
    type: str = Field(
        description="`cash` for a bank account, `credit` for a credit card"
    )
    current: int = Field(
        description="Posted balance in the currency's smallest unit (cents for "
        "USD), before pending transactions. Use this as the account's balance "
        "unless pending activity matters."
    )
    currency: str = Field(
        description="Three-letter ISO currency code in lowercase, e.g. usd"
    )
    available: dict[str, int] | None = Field(
        default=None,
        description="Cash accounts only: available funds, which unlike "
        "`current` count pending transactions, keyed by currency in the "
        'smallest unit, e.g. {"usd": 31005}. Null for credit accounts.',
    )
    used: dict[str, int] | None = Field(
        default=None,
        description="Credit accounts only: credit used, including pending "
        "charges, keyed by currency in the smallest unit. Null for cash "
        "accounts.",
    )
    as_of: Text = Field(
        default="",
        description="When Link last refreshed this balance (ISO 8601). It can "
        "be hours or days old.",
    )
