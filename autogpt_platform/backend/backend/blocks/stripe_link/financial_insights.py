"""
Stripe Link — financial insights blocks.

Read-only access to the bank accounts and cards a user shared through Link:
which accounts there are, their balances, and their transactions. Nothing here
moves money. Link offers this to US consumers only, on live accounts only
(there is no sandbox).

Accounts are shared when the user connects Stripe Link. A connection made
before the platform asked for that access is refused until it is reconnected.
"""

from enum import Enum
from typing import Any

from pydantic import field_validator, model_validator

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.stripe_link._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    StripeLinkCredentials,
    StripeLinkCredentialsField,
    StripeLinkCredentialsInput,
    link_api_request,
)
from backend.blocks.stripe_link._financial_api import (
    build_filters,
    iso_date,
    parse_account,
    parse_balance,
    parse_transaction,
    read_all,
    read_pages,
    unique_ids,
)
from backend.blocks.stripe_link._financial_models import (
    LinkBalance,
    LinkFinancialAccount,
    LinkTransaction,
)
from backend.blocks.stripe_link._financial_testdata import (
    TEST_ACCOUNT,
    TEST_BALANCE,
    TEST_BALANCE_WIRE,
    TEST_SOURCE_WIRE,
    TEST_TRANSACTION,
)
from backend.data.model import SchemaField

FINANCIAL_CREDENTIALS_DESCRIPTION = (
    "Connect your Stripe Link account and choose which bank accounts and cards "
    "the agent may read. Read-only: these blocks cannot move money."
)
SOURCE_IDS_DESCRIPTION = (
    "Only these accounts, by ID from List Financial Accounts. Leave empty for "
    "every account the user shared."
)

# Above Link's 100-per-request cap the transactions block reads further pages
# itself, up to this many records, so a month of activity fits in one run
# without a single run pulling an unbounded history.
MAX_TRANSACTIONS = 500


class TransactionOrigin(str, Enum):
    ALL = "all"
    EXTERNAL_CONNECTION = "external_connection"
    LINK = "link"


class StripeLinkListFinancialAccountsBlock(Block):
    """List the bank accounts and cards shared through Link (Link "sources")."""

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField(
            FINANCIAL_CREDENTIALS_DESCRIPTION
        )

    class Output(BlockSchemaOutput):
        accounts: list[LinkFinancialAccount] = SchemaField(
            description="Every bank account and card the user shared, with "
            "what each one is ready to provide"
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="e0c0c6e9-73bd-44c0-9174-dcab162ce362",
            description=(
                "List the bank accounts and credit cards the user shared "
                "through Stripe Link for financial insights, with each one's "
                "ID, name, last four digits and whether its balances and "
                "transactions are ready to read. Use the IDs to narrow List "
                "Transactions or Get Balances to particular accounts. To pay "
                "with Link, use List Payment Methods instead."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("accounts", [TEST_ACCOUNT])],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "data": [TEST_SOURCE_WIRE],
                    "has_more": False,
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            yield "accounts", await read_all(
                self._link_api_request,
                credentials,
                "/sources",
                [],
                parse=parse_account,
                cursor_of=lambda account: account.id,
            )
        except Exception as e:
            yield "error", str(e)


class StripeLinkListTransactionsBlock(Block):
    """Read transactions from the accounts shared through Link."""

    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField(
            FINANCIAL_CREDENTIALS_DESCRIPTION
        )
        start_date: str = SchemaField(
            description="Only transactions on or after this date, written "
            "YYYY-MM-DD. Leave empty for no lower bound.",
            placeholder="2026-06-01",
            default="",
        )
        end_date: str = SchemaField(
            description="Only transactions on or before this date, written "
            "YYYY-MM-DD. Leave empty for no upper bound.",
            placeholder="2026-06-30",
            default="",
        )
        source_ids: list[str] = SchemaField(
            description=SOURCE_IDS_DESCRIPTION, default_factory=list
        )
        origin: TransactionOrigin = SchemaField(
            description="`external_connection` for transactions from the "
            "user's connected banks and cards, `link` for purchases made "
            "through Link, `all` for both.",
            default=TransactionOrigin.ALL,
        )
        category: str = SchemaField(
            description="Only this spending category. Many transactions have "
            "no category, so leave this empty when a total must be complete.",
            default="",
            advanced=True,
        )
        limit: int = SchemaField(
            description=f"Most transactions to return, 1 to {MAX_TRANSACTIONS}. "
            "Above 100 the block reads further pages from Link itself.",
            default=100,
            ge=1,
            le=MAX_TRANSACTIONS,
        )
        starting_after: str = SchemaField(
            description="Continue after this transaction ID: pass `next_cursor` "
            "from the previous run and keep every other input the same.",
            default="",
            advanced=True,
        )

        _check_dates = field_validator("start_date", "end_date")(iso_date)
        _clean_ids = field_validator("source_ids")(unique_ids)

        @model_validator(mode="after")
        def _range_runs_forward(self):
            # YYYY-MM-DD compares correctly as text. Link would answer a
            # reversed range with an empty list, which reads as "no spending".
            if self.start_date and self.end_date and self.start_date > self.end_date:
                raise ValueError(
                    f"start_date {self.start_date} is after end_date {self.end_date}"
                )
            return self

    class Output(BlockSchemaOutput):
        transactions: list[LinkTransaction] = SchemaField(
            description="Matching transactions in the order Link returns them. "
            "Amounts are integers in the currency's smallest unit (cents for "
            "USD), negative for money leaving the account."
        )
        has_more: bool = SchemaField(
            description="True when more matching transactions exist beyond "
            "these. A total over a date range is incomplete until this is false."
        )
        next_cursor: str = SchemaField(
            description="Pass as `starting_after`, with the same filters, to "
            "read the next transactions. Empty when there are no more."
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="71f186bf-7935-4828-bd5e-9f276ef031e3",
            description=(
                "Read the user's transactions from the bank accounts and "
                "credit cards they shared through Stripe Link, filtered by "
                "date range, account or origin. Use it for questions about "
                "spending, income, merchants or subscriptions, such as how "
                "much they spent on dining last month. Amounts are integers "
                "in the smallest currency unit, negative for money leaving the "
                "account. When has_more is true, pass next_cursor back as "
                "starting_after to keep reading."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "start_date": "2026-06-01",
                "end_date": "2026-06-30",
                "source_ids": ["csmrpd_test"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("transactions", [TEST_TRANSACTION]),
                ("has_more", True),
                ("next_cursor", TEST_TRANSACTION.id),
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "data": [TEST_TRANSACTION.model_dump()],
                    "has_more": True,
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            origin = input_data.origin
            transactions, has_more = await read_pages(
                self._link_api_request,
                credentials,
                "/transactions",
                build_filters(
                    start_date=input_data.start_date,
                    end_date=input_data.end_date,
                    origin="" if origin == TransactionOrigin.ALL else origin.value,
                    category=input_data.category.strip(),
                    source_ids=input_data.source_ids,
                ),
                parse=parse_transaction,
                cursor_of=lambda transaction: transaction.id,
                max_items=input_data.limit,
                starting_after=input_data.starting_after.strip(),
            )
            yield "transactions", transactions
            yield "has_more", has_more
            yield "next_cursor", (
                transactions[-1].id if has_more and transactions else ""
            )
        except Exception as e:
            yield "error", str(e)


class StripeLinkGetBalancesBlock(Block):
    """Read current balances for the accounts shared through Link."""

    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField(
            FINANCIAL_CREDENTIALS_DESCRIPTION
        )
        source_ids: list[str] = SchemaField(
            description=SOURCE_IDS_DESCRIPTION, default_factory=list
        )

        _clean_ids = field_validator("source_ids")(unique_ids)

    class Output(BlockSchemaOutput):
        balances: list[LinkBalance] = SchemaField(
            description="One balance per account, in the smallest unit of that "
            "account's currency (cents for USD). Never add balances in "
            "different currencies together."
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="17869751-a981-4f51-b0c3-8ab486c72802",
            description=(
                "Get current balances for the bank accounts and credit cards "
                "the user shared through Stripe Link: the posted balance, plus "
                "available funds for bank accounts or credit used for cards. "
                "Amounts are integers in the smallest currency unit (cents for "
                "USD)."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("balances", [TEST_BALANCE])],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "data": [TEST_BALANCE_WIRE],
                    "has_more": False,
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            yield "balances", await read_all(
                self._link_api_request,
                credentials,
                "/balances",
                build_filters(source_ids=input_data.source_ids),
                parse=parse_balance,
                cursor_of=lambda balance: balance.source_id,
            )
        except Exception as e:
            yield "error", str(e)
