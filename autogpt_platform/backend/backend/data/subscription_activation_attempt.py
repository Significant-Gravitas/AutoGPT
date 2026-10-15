"""Durable confirmation intent; Stripe calls never share its commit boundary."""

from uuid import uuid4

from backend.data.db import query_raw_with_schema
from backend.data.subscription_activation_models import (
    ActivationAttempt,
    ActivationTerms,
    ActivationUnavailable,
)

_COLUMNS = """"id", "userId" AS user_id,
    "stripeSubscriptionId" AS subscription_id, "stripeCustomerId" AS customer_id,
    "terms", "returnTo" AS return_to, "confirmedAt" AS confirmed_at,
    "invoiceId" AS invoice_id"""


async def get_attempt(
    user_id: str, attempt_id: str | None = None
) -> ActivationAttempt | None:
    rows = await query_raw_with_schema(
        "SELECT " + _COLUMNS + ' FROM {schema_prefix}"ProActivationAttempt" '
        'WHERE "userId" = $1 AND ($2::text IS NULL OR "id" = $2)',
        user_id,
        attempt_id,
        model=ActivationAttempt,
    )
    return rows[0] if rows else None


async def save_quote(
    user_id: str,
    subscription_id: str,
    customer_id: str,
    terms: ActivationTerms,
    return_to: str,
) -> ActivationAttempt:
    rows = await query_raw_with_schema(
        """INSERT INTO {schema_prefix}"ProActivationAttempt"
           ("id", "userId", "stripeSubscriptionId", "stripeCustomerId", "terms",
            "returnTo", "status", "createdAt", "updatedAt")
           VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'quoted', NOW(), NOW())
           ON CONFLICT ("userId") DO UPDATE SET "terms" = EXCLUDED."terms",
               "returnTo" = EXCLUDED."returnTo", "updatedAt" = NOW()
           WHERE "ProActivationAttempt"."confirmedAt" IS NULL
             AND "ProActivationAttempt"."stripeSubscriptionId" = EXCLUDED."stripeSubscriptionId"
             AND "ProActivationAttempt"."stripeCustomerId" = EXCLUDED."stripeCustomerId"
           RETURNING """
        + _COLUMNS,
        str(uuid4()),
        user_id,
        subscription_id,
        customer_id,
        terms.model_dump_json(),
        return_to,
        model=ActivationAttempt,
    )
    if not rows:
        raise ActivationUnavailable("Activation confirmation is already in progress")
    return rows[0]


async def save_confirmation(attempt: ActivationAttempt) -> ActivationAttempt:
    rows = await query_raw_with_schema(
        """UPDATE {schema_prefix}"ProActivationAttempt"
           SET "confirmedAt" = COALESCE("confirmedAt", NOW()),
               "status" = 'processing', "updatedAt" = NOW()
           WHERE "id" = $1 AND "userId" = $2
             AND "terms" = $3::jsonb AND "stripeSubscriptionId" = $4
             AND "stripeCustomerId" = $5 RETURNING """
        + _COLUMNS,
        attempt.id,
        attempt.user_id,
        attempt.terms.model_dump_json(),
        attempt.subscription_id,
        attempt.customer_id,
        model=ActivationAttempt,
    )
    if not rows:
        raise ActivationUnavailable("The terms changed. Review a fresh preview.")
    return rows[0]
