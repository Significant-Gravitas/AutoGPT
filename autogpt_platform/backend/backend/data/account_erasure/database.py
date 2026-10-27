"""The account's rows, last, in one transaction.

Most tables cascade from `User`. This deletes, before it, the rows that would
block that delete or survive it: rows with a user ID and no foreign key, the
orgs only this account belongs to, and leftover webhook rows. The auth
identity goes in the same transaction as the `User` row, so the
orphan-healing job (`heal_orphaned_auth_identities`) never sees an identity
without its account and re-creates it.

Kept on purpose:
- The credit ledger (`CreditTransaction`), a billing record kept for six
  years, with its user ID: `ledger.py` strips the personal content from its
  metadata, in the same transaction.
- `PlatformCostLog` (our provider costs): the foreign key nulls its user ID.
- `SubscriptionTrialClaim`: hashes that stop a re-signup from claiming a
  second trial; they carry no account ID.
- `AuditLog` rows of shared orgs the account acted in.
"""

from datetime import timedelta

from pydantic import BaseModel

from backend.data.account_erasure.ledger import count_unscrubbed, scrub_ledger
from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.data.db import execute_raw_with_schema, query_raw_with_schema, transaction
from backend.data.user import _invalidate_user_caches

# Executions hold most of an active account's rows (node executions and their
# inputs and outputs), so they go first, in batches, outside the transaction.
# Only its own: an agent another account has run went to the "deleted user"
# account (`shared_agents.py`) with those runs.
_EXECUTIONS = """
    FROM {schema_prefix}"AgentGraphExecution" e
   WHERE e."userId" = $1
"""
_EXECUTION_BATCH = 200
_TRANSACTION_TIMEOUT = timedelta(minutes=5)


SqlArg = str | list[str]


class Deletion(BaseModel):
    label: str
    table: str
    condition: str
    args: list[SqlArg]


async def delete_rows(subject: Subject, *, dry_run: bool) -> StepReport:
    deletions = planned_deletions(subject)
    if dry_run:
        counts = {
            "executions": await _count(_EXECUTIONS, [subject.user_id]),
            "credit_transactions_scrubbed": await count_unscrubbed(subject.user_id),
        }
        for deletion in deletions:
            counts[deletion.label] = await _count(
                f'FROM {{schema_prefix}}"{deletion.table}" WHERE {deletion.condition}',
                deletion.args,
            )
        return StepReport(name="database", status=StepStatus.WOULD_DO, counts=counts)

    counts = {"executions": await _delete_executions(subject.user_id)}
    async with transaction(timeout=_TRANSACTION_TIMEOUT) as tx:
        counts["credit_transactions_scrubbed"] = await scrub_ledger(
            subject.user_id, client=tx
        )
        for deletion in deletions:
            counts[deletion.label] = await execute_raw_with_schema(
                f'DELETE FROM {{schema_prefix}}"{deletion.table}" '
                f"WHERE {deletion.condition}",
                *deletion.args,
                client=tx,
            )
    _invalidate_user_caches(subject.user_id, subject.email)
    return StepReport(name="database", status=StepStatus.DONE, counts=counts)


def planned_deletions(subject: Subject) -> list[Deletion]:
    """In order: everything that references the account or its sole orgs
    without cascading, then the orgs, the auth identity and the account."""
    return [
        *_by_user(subject.user_id, _BEFORE_ORGS),
        *_invitation_deletions(subject),
        *_sole_org_deletions(subject.sole_org_ids),
        *_by_user(subject.user_id, _AFTER_ORGS),
        _verification_deletion(subject),
        *_by_user(subject.user_id, _LAST),
    ]


# (label, table, condition on $1 = the user ID)
_BEFORE_ORGS = [
    ("kv_data", "AgentNodeExecutionKeyValueData", '"userId" = $1'),
    ("refund_requests", "CreditRefundRequest", '"userId" = $1'),
    ("private_embeddings", "UnifiedContentEmbedding", '"userId" = $1'),
    # The account's own stored credentials in shared orgs; those in its sole
    # orgs go with the org. Team and org credentials stay.
    (
        "personal_credentials",
        "IntegrationCredential",
        '"ownerType" = \'USER\' AND "ownerId" = $1',
    ),
]
_AFTER_ORGS = [
    # These would cascade from the account too, but they reference its agent
    # graphs and library folders with ON DELETE RESTRICT; deleting them first
    # keeps the cascade from depending on trigger order.
    ("library_agents", "LibraryAgent", '"userId" = $1'),
    ("presets", "AgentPreset", '"userId" = $1'),
    ("webhooks", "IntegrationWebhook", '"userId" = $1'),
]
_LAST = [
    # Cascades to its sessions and linked sign-in accounts.
    ("auth_identity", "UserAuthIdentity", '"id" = $1'),
    ("user", "User", '"id" = $1'),
]


def _by_user(user_id: str, rows: list[tuple[str, str, str]]) -> list[Deletion]:
    return [
        Deletion(label=label, table=table, condition=condition, args=[user_id])
        for label, table, condition in rows
    ]


def _invitation_deletions(subject: Subject) -> list[Deletion]:
    """Invitations to this account into other orgs and teams, by ID or by
    the email they were sent to."""
    condition = '"targetUserId" = $1'
    args: list[SqlArg] = [subject.user_id]
    if subject.email:
        condition += ' OR LOWER("email") = LOWER($2)'
        args.append(subject.email)
    return [
        Deletion(label=label, table=table, condition=condition, args=args)
        for label, table in (
            ("org_invitations", "OrgInvitation"),
            ("team_invitations", "TeamInvite"),
        )
    ]


def _verification_deletion(subject: Subject) -> Deletion:
    """Better Auth's verification rows of the account: reset links hold its
    ID as the value, and the sign-up and verification email cooldowns
    (`auth-email-cooldown.ts`) name its address after the kind, as
    `verify-email:<email>`. The cooldowns expire within minutes, but carry
    the email until Better Auth clears them."""
    condition = '"value" = $1'
    args: list[SqlArg] = [subject.user_id]
    if subject.email:
        condition += (
            ' OR RIGHT(LOWER("identifier"), LENGTH($2::text) + 1) '
            "= ':' || LOWER($2::text)"
        )
        args.append(subject.email)
    return Deletion(
        label="auth_verifications",
        table="UserAuthVerification",
        condition=condition,
        args=args,
    )


def _sole_org_deletions(org_ids: list[str]) -> list[Deletion]:
    """Orgs only this account belongs to. Their ledger and transfer rows would
    refuse the delete; everything else of theirs (teams, members, stored
    integration credentials, balance, profile) cascades."""
    if not org_ids:
        return []
    ids: list[SqlArg] = [org_ids]
    return [
        Deletion(
            label="org_credit_transactions",
            table="OrgCreditTransaction",
            condition='"orgId" = ANY($1::text[])',
            args=ids,
        ),
        Deletion(
            label="org_transfer_requests",
            table="TransferRequest",
            condition='"sourceOrganizationId" = ANY($1::text[]) '
            'OR "targetOrganizationId" = ANY($1::text[])',
            args=ids,
        ),
        Deletion(
            label="org_audit_logs",
            table="AuditLog",
            condition='"organizationId" = ANY($1::text[])',
            args=ids,
        ),
        Deletion(
            label="organizations",
            table="Organization",
            condition='"id" = ANY($1::text[])',
            args=ids,
        ),
    ]


async def _delete_executions(user_id: str) -> int:
    deleted = 0
    while True:
        batch = await execute_raw_with_schema(
            'DELETE FROM {schema_prefix}"AgentGraphExecution" WHERE "id" IN '
            f'(SELECT e."id" {_EXECUTIONS} LIMIT {_EXECUTION_BATCH})',
            user_id,
        )
        deleted += batch
        if batch < _EXECUTION_BATCH:
            return deleted


async def _count(from_where: str, args: list[SqlArg]) -> int:
    rows = await query_raw_with_schema(f"SELECT COUNT(*)::int AS n {from_where}", *args)
    return int(rows[0]["n"]) if rows else 0
