"""The account's credit ledger stays: it is a billing record, kept for six
years. Amounts, types, dates, running balances and Stripe IDs stay with the
user ID; the metadata is cut down to IDs and amounts (`LEDGER_KEYS`), which
drops the block inputs, free-text reasons, and the names, emails and
addresses in stored copies of Stripe objects (checkout sessions, refunds,
disputes).
"""

from prisma import Prisma

from backend.data.db import execute_raw_with_schema, query_raw_with_schema

# Kept when the value is a scalar, or as the `id` of a nested Stripe object
# (an expanded `payment_intent`, say). Everything else goes.
LEDGER_KEYS = [
    # Usage: which run and block spent the credits.
    "graph_exec_id",
    "graph_id",
    "chat_session_id",
    "node_id",
    "node_exec_id",
    "block_id",
    "block",
    # Grants: the admin who gave them.
    "admin_id",
    # Stripe: the payment, refund, dispute or invoice, and its amounts.
    "id",
    "object",
    "payment_intent",
    "charge",
    "invoice",
    "subscription",
    "customer",
    "stripe_customer_id",
    "stripe_subscription_id",
    "stripe_invoice_id",
    "billing_reason",
    "amount",
    "amount_total",
    "amount_subtotal",
    "amount_received",
    "amount_refunded",
    "currency",
    "status",
    "payment_status",
    "created",
    "livemode",
    "mode",
]

# $2 = `LEDGER_KEYS`. Metadata that is not an object becomes NULL.
# jsonb_build_object() rather than a '{}' literal: the template goes through
# str.format for {schema_prefix}.
_SCRUBBED = """
    CASE WHEN jsonb_typeof(ct."metadata") = 'object' THEN (
        SELECT COALESCE(
                 jsonb_object_agg(m.key, CASE
                   WHEN jsonb_typeof(m.value) = 'object' THEN m.value -> 'id'
                   ELSE m.value END),
                 jsonb_build_object())
          FROM jsonb_each(ct."metadata") m
         WHERE m.key = ANY($2::text[])
           AND (jsonb_typeof(m.value) IN ('string', 'number', 'boolean')
                OR (jsonb_typeof(m.value) = 'object'
                    AND jsonb_typeof(m.value -> 'id') = 'string'))
    ) END
"""
# $1 = the account. Only rows not scrubbed yet, so a rerun changes nothing.
_UNSCRUBBED = f"""
     WHERE ct."userId" = $1
       AND ct."metadata" IS DISTINCT FROM ({_SCRUBBED})
"""
_COUNT = f"""
    SELECT COUNT(*)::int AS n FROM {{schema_prefix}}"CreditTransaction" ct
    {_UNSCRUBBED}
"""
_SCRUB = f"""
    UPDATE {{schema_prefix}}"CreditTransaction" ct SET "metadata" = ({_SCRUBBED})
    {_UNSCRUBBED}
"""


async def count_unscrubbed(user_id: str) -> int:
    rows = await query_raw_with_schema(_COUNT, user_id, LEDGER_KEYS)
    return int(rows[0]["n"]) if rows else 0


async def scrub_ledger(user_id: str, *, client: Prisma) -> int:
    """Strip the personal content from the account's ledger rows; returns
    how many changed."""
    return await execute_raw_with_schema(_SCRUB, user_id, LEDGER_KEYS, client=client)
