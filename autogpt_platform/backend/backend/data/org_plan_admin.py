"""What an AutoGPT admin does for a sales-led team plan.

Sales invoices the customer outside the product; an admin then sets the team
organization's plan (tier, seats, end date, status, note) and adds the credits
the invoice bought to its wallet. Every change writes an ``AuditLog`` row
whose actor is the admin, and a plan change reaches every process at once
(``forget_org_plan``). Credits are keyed by the invoice or other reference, so
adding them twice with one reference adds them once.
"""

import logging
from datetime import datetime
from typing import Any

import prisma.types
from prisma import Prisma
from prisma.enums import CreditTransactionType
from prisma.models import AuditLog, Organization, OrgCreditTransaction
from pydantic import BaseModel, Field, field_validator

from backend.copilot.rate_limit import SubscriptionTier
from backend.data.db import query_raw_with_schema, transaction
from backend.data.org_plan import (
    PLAN_TIERS,
    OrgPlan,
    PlanStatus,
    forget_org_plan,
    plan_from_row,
)
from backend.data.org_plan_summary import OrgPlanSummary, org_plan_summary
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

PLAN_AUDIT_ENTITY = "OrganizationSubscription"
CREDIT_AUDIT_ENTITY = "OrgCreditGrant"
SALES_PLAN_CODE = "sales"


class OrgPlanInput(BaseModel):
    tier: SubscriptionTier
    seats: int = Field(ge=1, le=100_000)
    ends_at: datetime | None = Field(
        default=None, description="When the plan stops counting; none runs on."
    )
    status: PlanStatus = "active"
    note: str | None = Field(default=None, max_length=500)

    @field_validator("tier")
    @classmethod
    def _a_paid_tier(cls, tier: SubscriptionTier) -> SubscriptionTier:
        if tier not in PLAN_TIERS:
            raise ValueError(
                f"A plan gives one of {', '.join(t.value for t in PLAN_TIERS)}"
            )
        return tier


class OrgCreditGrantResult(BaseModel):
    org_id: str
    reference: str
    # False when credits with this reference were already added.
    granted: bool
    balance: int


async def set_org_plan(
    org_id: str, admin_user_id: str, plan: OrgPlanInput
) -> OrgPlanSummary:
    """Create or replace a team organization's plan."""
    await _team_org(org_id)
    fields: dict[str, Any] = {
        "planCode": SALES_PLAN_CODE,
        "planTier": plan.tier.value,
        "seats": plan.seats,
        "endsAt": plan.ends_at,
        "status": plan.status,
        "note": (plan.note or "").strip() or None,
    }
    async with transaction() as tx:
        before, after = await _write_plan(tx, org_id, fields)
        await _audit(
            tx,
            admin_user_id,
            org_id,
            entity=PLAN_AUDIT_ENTITY,
            action="org_plan.set",
            before=_plan_json(before),
            after=_plan_json(after),
        )
    forget_org_plan(org_id)
    logger.info(f"Admin {admin_user_id} set the plan of org {org_id}: {fields}")
    return await org_plan_summary(org_id)


async def grant_org_credits(
    org_id: str, admin_user_id: str, amount: int, reference: str, note: str | None
) -> OrgCreditGrantResult:
    """Add *amount* credits to a team organization's wallet, once per reference."""
    await _team_org(org_id)
    reference = reference.strip()
    metadata = {"reason": note or "Team plan credits", "reference": reference}
    async with transaction() as tx:
        balance = await _grant_once(tx, org_id, admin_user_id, amount, metadata)
        if balance is None:
            return OrgCreditGrantResult(
                org_id=org_id,
                reference=reference,
                granted=False,
                balance=await _lock_balance(tx, org_id),
            )
        await _audit(
            tx,
            admin_user_id,
            org_id,
            entity=CREDIT_AUDIT_ENTITY,
            action="org_credits.grant",
            before=None,
            after={**metadata, "amount": amount, "balance": balance},
        )
    logger.info(
        f"Admin {admin_user_id} added {amount} credits to org {org_id} ({reference})"
    )
    return OrgCreditGrantResult(
        org_id=org_id, reference=reference, granted=True, balance=balance
    )


async def list_account_team_orgs(user_id: str) -> list[OrgPlanSummary]:
    """The live team organizations *user_id* is an active member of."""
    orgs = await Organization.prisma().find_many(
        where={
            "isPersonal": False,
            "deletedAt": None,
            "Members": {"some": {"userId": user_id, "status": "ACTIVE"}},
        },
        order={"name": "asc"},
    )
    return [await org_plan_summary(org.id) for org in orgs]


async def _team_org(org_id: str) -> Organization:
    org = await Organization.prisma().find_unique(where={"id": org_id})
    if org is None or org.deletedAt is not None:
        raise NotFoundError(f"Organization #{org_id} not found")
    if org.isPersonal:
        raise ValueError(
            "A personal organization has no plan of its own: it uses its owner's. "
            "Set the owner's tier or credits on the Users page instead."
        )
    return org


async def _write_plan(
    tx: Prisma, org_id: str, fields: dict[str, Any]
) -> tuple[OrgPlan | None, OrgPlan | None]:
    """Upsert the plan row; the plan before and after."""
    before = await tx.organizationsubscription.find_unique(
        where={"organizationId": org_id}
    )
    row = await tx.organizationsubscription.upsert(
        where={"organizationId": org_id},
        data={"create": {"organizationId": org_id, **fields}, "update": fields},
    )
    return plan_from_row(before), plan_from_row(row)


async def _grant_once(
    tx: Prisma,
    org_id: str,
    admin_user_id: str,
    amount: int,
    metadata: dict[str, str],
) -> int | None:
    """Add the credits and record them under the reference's key; None, adding
    nothing, when that reference was granted before."""
    key = f"admin-grant:{metadata['reference']}"
    # Locks the wallet row, so two grants with one reference can't both find
    # it unused.
    await _lock_balance(tx, org_id)
    if await _granted_before(tx, org_id, key):
        return None
    balance = await _add_to_balance(tx, org_id, amount)
    await OrgCreditTransaction.prisma(tx).create(
        data={
            "transactionKey": key,
            "orgId": org_id,
            "amount": amount,
            "type": CreditTransactionType.GRANT,
            "runningBalance": balance,
            "initiatedByUserId": admin_user_id,
            "metadata": SafeJson({**metadata, "admin_id": admin_user_id}),
        }
    )
    return balance


async def _lock_balance(tx: Prisma, org_id: str) -> int:
    rows = await query_raw_with_schema(
        'INSERT INTO {schema_prefix}"OrgBalance" ("orgId", "balance", "updatedAt") '
        "VALUES ($1, 0, NOW()) "
        'ON CONFLICT ("orgId") DO UPDATE SET "updatedAt" = {schema_prefix}"OrgBalance"."updatedAt" '
        'RETURNING "balance"',
        org_id,
        client=tx,
    )
    return rows[0]["balance"]


async def _add_to_balance(tx: Prisma, org_id: str, amount: int) -> int:
    rows = await query_raw_with_schema(
        'UPDATE {schema_prefix}"OrgBalance" SET "balance" = "balance" + $2, '
        '"updatedAt" = NOW() WHERE "orgId" = $1 RETURNING "balance"',
        org_id,
        amount,
        client=tx,
    )
    return rows[0]["balance"]


async def _granted_before(tx: Prisma, org_id: str, key: str) -> bool:
    existing = await OrgCreditTransaction.prisma(tx).find_unique(
        where={
            "orgCreditTransactionIdentifier": {"transactionKey": key, "orgId": org_id}
        }
    )
    return existing is not None


def _plan_json(plan: OrgPlan | None) -> dict[str, Any] | None:
    return plan.model_dump(mode="json", exclude={"updated_at"}) if plan else None


async def _audit(
    tx: Prisma,
    admin_user_id: str,
    org_id: str,
    *,
    entity: str,
    action: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> None:
    data: prisma.types.AuditLogCreateInput = {
        "organizationId": org_id,
        "actorUserId": admin_user_id,
        "entityType": entity,
        "entityId": org_id,
        "action": action,
    }
    if before is not None:
        data["beforeJson"] = SafeJson(before)
    if after is not None:
        data["afterJson"] = SafeJson(after)
    await AuditLog.prisma(tx).create(data=data)
