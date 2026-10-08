"""The role backfill's query against the real tables: the pick the wizard kept
and the role in the business understanding, side by side, for every account
that has either."""

import uuid

import prisma.models
import pytest

from backend.cli import onboarding_role_backfill as cli
from backend.data.onboarding_role import OnboardingRole, save_onboarding_role
from backend.util.json import SafeJson


@pytest.fixture
async def account():
    user_id = str(uuid.uuid4())
    await prisma.models.User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.com"}
    )
    yield user_id
    await prisma.models.User.prisma().delete(where={"id": user_id})


@pytest.mark.asyncio
async def test_both_records_of_the_role_are_read(account):
    await save_onboarding_role(account, OnboardingRole(choice="Other", other="CFO"))
    await prisma.models.CoPilotUnderstanding.prisma().create(
        data={
            "userId": account,
            "data": SafeJson({"name": "Sam", "business": {"user_role": "Marketing"}}),
        }
    )

    (record,) = [r for r in await cli._records() if r.user_id == account]

    assert record.email == f"{account}@example.com"
    assert (record.choice, record.other) == ("Other", "CFO")
    assert record.understanding_role == "Marketing"
    assert record.marketing_opt_out_at is None


@pytest.mark.asyncio
async def test_an_account_with_no_role_is_not_read(account):
    assert account not in {r.user_id for r in await cli._records()}


@pytest.mark.asyncio
async def test_the_country_a_checkout_recorded_is_read(account):
    await save_onboarding_role(account, OnboardingRole(choice="Marketing"))
    await prisma.models.User.prisma().update(
        where={"id": account},
        data={"stripeCustomerId": f"cus_{account}", "marketingExcludedCountry": "RU"},
    )

    (record,) = [r for r in await cli._records() if r.user_id == account]

    assert record.stripe_customer_id == f"cus_{account}"
    assert record.excluded_country == "RU"
