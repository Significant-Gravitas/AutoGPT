"""One paid Stripe checkout credits the balance once, however many fulfilments
race for it (GHSA-m4qg-8hhm-9w94).

`PATCH /credits` can be fired in a burst and Stripe sends two events per
payment, so several `fulfill_checkout` calls for the same session run at
once. Each passes the unlocked pre-check, then runs the activation CTE. That
CTE locked only the `UserBalance` row, so the loser waited for the winner,
re-read the balance the winner had just raised, and added the amount again
from its own statement-start snapshot of the still-inactive transaction row.
"""

import asyncio

import pytest
import stripe
from prisma import Prisma
from prisma.enums import CreditTransactionType
from prisma.models import CreditTransaction, User, UserBalance

from backend.data import credit as credit_module
from backend.data import db
from backend.data.credit import UserCredit
from backend.util.test import SpinTestServer

USER_ID = "fulfill-race-test-user"
SESSION_ID = "cs_test_fulfill_race"
PAYMENT_INTENT_ID = "pi_test_fulfill_race"
AMOUNT = 1000
WAIT_SECONDS = 15


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_fulfilments_of_one_checkout_credit_it_once(
    server: SpinTestServer, mocker
):
    """The bad order is forced, not hoped for: the first fulfilment runs its
    activation inside a transaction it holds open until the database reports
    the second fulfilment's statement waiting on the lock. The second has
    then taken its snapshot before the first commits, so it only ever sees
    the transaction row inactive; the row lock has to make it a no-op."""
    await _reset_user(balance=0)
    await CreditTransaction.prisma().create(
        data={
            "userId": USER_ID,
            "amount": AMOUNT,
            "type": CreditTransactionType.TOP_UP,
            "transactionKey": SESSION_ID,
            "isActive": False,
        }
    )
    mocker.patch(
        "stripe.checkout.Session.retrieve_async",
        new=mocker.AsyncMock(return_value=_paid_session()),
    )
    # The second fulfilment runs on its own client, i.e. its own query engine
    # and connection, as a second API replica or webhook worker would: on one
    # client the engine runs the two statements one after the other.
    other_client = Prisma(auto_register=False, datasource={"url": db.DATABASE_URL})
    await other_client.connect()
    forced = _ForcedOrder(credit_module.query_raw_with_schema, other_client)
    # A bound `async def`, not the instance: `patch` only awaits a side effect
    # that `inspect.iscoroutinefunction` recognises.
    mocker.patch.object(credit_module, "query_raw_with_schema", side_effect=forced.run)
    credit_system = UserCredit()
    try:
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                credit_system.fulfill_checkout(session_id=SESSION_ID),
                credit_system.fulfill_checkout(session_id=SESSION_ID),
                return_exceptions=True,
            ),
            timeout=WAIT_SECONDS * 3,
        )
        assert [o for o in outcomes if isinstance(o, BaseException)] == []

        balance = await UserBalance.prisma().find_unique(where={"userId": USER_ID})
        assert balance is not None and balance.balance == AMOUNT
        activated = await CreditTransaction.prisma().find_many(
            where={"userId": USER_ID, "isActive": True}
        )
        assert [
            (row.transactionKey, row.amount, row.runningBalance) for row in activated
        ] == [(PAYMENT_INTENT_ID, AMOUNT, AMOUNT)]
        assert (
            await CreditTransaction.prisma().count(
                where={"userId": USER_ID, "isActive": False}
            )
            == 0
        )
    finally:
        await other_client.disconnect()
        await _reset_user(balance=None)


class _ForcedOrder:
    """Stands in for `query_raw_with_schema` for the two racing activations.

    The first call runs inside a transaction held open until the second
    call's statement is waiting on a lock, then commits. The second call runs
    plainly once the first holds its locks, so it blocks behind them."""

    def __init__(self, real_query, other_client: Prisma):
        self._real_query = real_query
        self._other_client = other_client
        self._calls = 0
        self._first_holds_its_locks = asyncio.Event()

    async def run(self, *args, **kwargs):
        self._calls += 1
        if self._calls > 1:
            await asyncio.wait_for(
                self._first_holds_its_locks.wait(), timeout=WAIT_SECONDS
            )
            result = await self._real_query(*args, client=self._other_client, **kwargs)
            return result
        async with db.transaction() as tx:
            result = await self._real_query(*args, client=tx, **kwargs)
            self._first_holds_its_locks.set()
            await asyncio.wait_for(
                _until_a_statement_waits_on_a_lock(tx), timeout=WAIT_SECONDS
            )
        return result


async def _until_a_statement_waits_on_a_lock(tx) -> None:
    while True:
        # Inside a transaction, pg_stat_activity is frozen at first access
        # until the snapshot is cleared, so clear it before every look.
        await tx.execute_raw("SELECT pg_stat_clear_snapshot()")
        # `query` is cut at track_activity_query_size (1 KB by default) and
        # starts with the statement's leading whitespace, so match a name
        # from its first line.
        rows = await tx.query_raw(
            "SELECT count(*)::int AS waiting FROM pg_stat_activity "
            "WHERE wait_event_type = 'Lock' AND pid <> pg_backend_pid() "
            "AND datname = current_database() "
            "AND query LIKE '%user_balance_lock%'"
        )
        if rows and rows[0]["waiting"] >= 1:
            return
        await asyncio.sleep(0.05)


async def _reset_user(balance: int | None) -> None:
    await CreditTransaction.prisma().delete_many(where={"userId": USER_ID})
    await UserBalance.prisma().delete_many(where={"userId": USER_ID})
    await User.prisma().delete_many(where={"id": USER_ID})
    if balance is None:
        return
    await User.prisma().create(
        data={"id": USER_ID, "email": f"{USER_ID}@example.com", "name": "Race"}
    )
    await UserBalance.prisma().create(data={"userId": USER_ID, "balance": balance})


def _paid_session() -> stripe.checkout.Session:
    return stripe.checkout.Session.construct_from(
        {
            "id": SESSION_ID,
            "object": "checkout.session",
            "payment_status": "paid",
            "amount_total": AMOUNT,
            "currency": "usd",
            "payment_intent": {
                "id": PAYMENT_INTENT_ID,
                "object": "payment_intent",
            },
        },
        "sk_test",
    )
