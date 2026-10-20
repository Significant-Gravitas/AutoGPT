"""Global seat cap for concurrent introductory trials.

LaunchDarkly can hold the *number*, but not enforce it: a cap on how many
people are on trial at once is a count over our own rows, so the count and
the reservation have to happen together, here.

A seat is occupied by either:

* a **running trial** -- exactly the rows ``trial_is_active`` calls active, so
  the cap counts the same population the product calls "on trial"; or
* a **checkout opened recently** -- a card screen someone is standing in
  front of right now. Without this the cap would be enforced against a count
  that ignores everyone mid-enrolment, and a burst of simultaneous checkouts
  would all be admitted.

Seats are taken under an advisory lock and released by time, never by
cleanup: an abandoned checkout stops counting once ``CHECKOUT_RESERVATION``
passes. The window is deliberately much shorter than Stripe's 24h session
expiry -- a day-long hold would let abandoned carts starve a small cap --
which means someone who leaves checkout open past the window can still
complete it in that tab, pushing the count one over. Coming back through our
checkout endpoint instead re-admits them under the lock
(:func:`renew_trial_seat`), so returners cannot all pass on one free seat. The cap is a throttle on
enrolment, not an invariant, and it is enforced *before* the card screen so
that going over costs nothing: no one is ever turned away after paying.
"""

from datetime import timedelta

from prisma import Prisma
from pydantic import BaseModel

from backend.data.db import execute_raw_with_schema, query_raw_with_schema, transaction
from backend.data.subscription_trial_config import TrialOffer

# How long an opened checkout holds a seat. Long enough to type in a card,
# short enough that abandoned ones free up without a sweeper.
CHECKOUT_RESERVATION = timedelta(minutes=30)

_CAPACITY_LOCK = "trial-capacity"

TRIAL_FULL = "The trial is full right now. Please check back later."

# One definition of "holding a seat", so the cap and the holder check can
# never drift apart. $2 is the reservation window.
_SEAT_HELD = """
    -- running: mirrors trial_is_active()
    ("status" = 'trialing' AND "cardVerifiedAt" IS NOT NULL
     AND "endsAt" > NOW())
    -- or standing at a checkout opened within the reservation window
    OR ("status" = 'checkout_pending' AND "consumedAt" IS NULL
        AND "updatedAt" > NOW() - $2::interval)
"""

# Row filters, both constants -- $1 is always a bound parameter.
_EVERY_OTHER_TRIAL = "($1 = '' OR \"id\" <> $1)"
_ONLY_THIS_TRIAL = '"id" = $1'


class TrialCapacityReached(ValueError):
    """No seat is free under the offer's cap."""


class _Seats(BaseModel):
    seats: int


async def trial_seat_available(
    offer: TrialOffer,
    *,
    trial_id: str | None = None,
    client: Prisma | None = None,
) -> bool:
    """May the enrolment ``trial_id`` occupy a seat under *offer*'s cap?

    True when a seat is free, and also when this enrolment already holds
    one -- otherwise a reservation would be revoked by whoever filled the
    remaining seats while its owner was still typing in a card.

    Callers about to *take* a seat must hold :func:`lock_trial_capacity` on
    the same transaction; callers merely deciding whether to show the offer
    need not, and should tolerate the answer being a moment stale.
    """
    if offer.max_active_trials is None:
        return True
    if offer.max_active_trials == 0:
        # A hard pause on enrolment. Seats already held -- running trials and
        # open checkouts alike -- are kept, but nothing new gets one.
        return bool(trial_id) and await _holds_seat(trial_id, client=client)
    held_by_others = await count_trial_seats(exclude_trial_id=trial_id, client=client)
    if held_by_others < offer.max_active_trials:
        return True
    return bool(trial_id) and await _holds_seat(trial_id, client=client)


async def renew_trial_seat(offer: TrialOffer, trial_id: str) -> bool:
    """Re-admit the checkout ``trial_id`` under *offer*'s cap and restart its hold.

    A checkout whose hold has lapsed no longer holds a seat, so returners
    checked without the lock could each pass on the same free seat. The check
    and the renewal happen together under the capacity lock, in their own
    short transaction, before any Stripe call.
    """
    if offer.max_active_trials is None:
        return True
    async with transaction() as tx:
        await lock_trial_capacity(tx)
        if not await trial_seat_available(offer, trial_id=trial_id, client=tx):
            return False
        await execute_raw_with_schema(
            'UPDATE {schema_prefix}"SubscriptionTrial" SET "updatedAt" = NOW() '
            'WHERE "id" = $1 AND "status" = \'checkout_pending\'',
            trial_id,
            client=tx,
        )
    return True


async def lock_trial_capacity(tx: Prisma) -> None:
    """Serialise seat-taking for the duration of *tx*.

    Held by every caller that counts seats with intent to take one, so two
    simultaneous enrolments cannot both read the last free seat.
    """
    await query_raw_with_schema(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))::text",
        _CAPACITY_LOCK,
        client=tx,
    )


async def count_trial_seats(
    *, exclude_trial_id: str | None = None, client: Prisma | None = None
) -> int:
    """Seats currently held, optionally ignoring one enrolment's own seat.

    ``exclude_trial_id`` is what lets a user who already holds a seat resume
    their own checkout when the trial is otherwise full.
    """
    return await _count_seats(_EVERY_OTHER_TRIAL, exclude_trial_id or "", client=client)


async def _holds_seat(trial_id: str, *, client: Prisma | None = None) -> bool:
    return bool(await _count_seats(_ONLY_THIS_TRIAL, trial_id, client=client))


async def _count_seats(
    row_filter: str, target: str, *, client: Prisma | None = None
) -> int:
    rows = await query_raw_with_schema(
        f'SELECT COUNT(*)::int AS seats FROM {{schema_prefix}}"SubscriptionTrial" '
        f"WHERE {row_filter} AND ({_SEAT_HELD})",
        target,
        f"{CHECKOUT_RESERVATION.total_seconds()} seconds",
        model=_Seats,
        client=client,
    )
    return rows[0].seats if rows else 0
