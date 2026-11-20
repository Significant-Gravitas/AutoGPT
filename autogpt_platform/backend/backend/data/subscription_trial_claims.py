"""Durable introductory-offer claims, independent of account deletion."""

import json
from hashlib import sha256

from prisma import Prisma
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.data.subscription_trial import TrialState


class LinkedIdentity(BaseModel):
    provider: str
    subject: str


def trial_identity_key(kind: str, value: str) -> str:
    payload = json.dumps([kind, value], separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


async def claim_trial_identities(
    trial: TrialState, fingerprint: str | None, tx: Prisma
) -> bool:
    if not fingerprint:
        return False
    identities = await query_raw_with_schema(
        'SELECT "providerId" AS provider, "accountId" AS subject '
        'FROM {schema_prefix}"UserAuthAccount" '
        'WHERE "userId" = $1 AND "providerId" <> \'credential\'',
        trial.user_id,
        model=LinkedIdentity,
        client=tx,
    )
    keys = sorted(
        {
            trial_identity_key("card", fingerprint),
            trial_identity_key("user", trial.user_id),
            trial_identity_key("customer", trial.customer_id),
            *(
                trial_identity_key(f"provider:{identity.provider}", identity.subject)
                for identity in identities
            ),
        }
    )
    return await _claim_keys(trial.id, keys, tx)


async def _claim_keys(trial_id: str, keys: list[str], tx: Prisma) -> bool:
    for key in keys:
        await query_raw_with_schema(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))::text",
            f"intro-offer:{key}",
            client=tx,
        )
    existing = await tx.subscriptiontrialclaim.find_many(where={"key": {"in": keys}})
    if any(claim.trialId != trial_id for claim in existing):
        return False
    await tx.subscriptiontrialclaim.create_many(
        data=[{"key": key, "trialId": trial_id} for key in keys],
        skip_duplicates=True,
    )
    return True
