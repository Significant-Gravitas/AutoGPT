"""The role backfill's query against the real tables: the pick the wizard kept,
its draft and the role in the business understanding, side by side, for every
account that has any of them."""

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


def _draft(**fields) -> dict:
    return {
        "version": 1,
        "currentStep": "subscription",
        "completedSteps": ["team", "autopilot", "role", "painPoints", "hire"],
        "role": "Marketing",
        "otherRole": "",
        "painPoints": [],
        "otherPainPoint": "",
        "selectedBilling": "monthly",
        "hasUserSelectedBilling": False,
        "selectedCountryCode": "US",
        "hiredTemplateIds": [],
        **fields,
    }


@pytest.mark.asyncio
async def test_the_draft_of_someone_who_stopped_at_the_paywall_is_read(account):
    await prisma.models.UserOnboarding.prisma().create(
        data={"userId": account, "wizardProgress": SafeJson(_draft())}
    )

    (record,) = [r for r in await cli._records() if r.user_id == account]

    assert record.choice is None
    assert record.reliable_role() == OnboardingRole(choice="Marketing")


@pytest.mark.asyncio
async def test_a_draft_still_on_the_role_step_is_not_read(account):
    await prisma.models.UserOnboarding.prisma().create(
        data={
            "userId": account,
            "wizardProgress": SafeJson(
                _draft(currentStep="role", completedSteps=["team", "autopilot"])
            ),
        }
    )

    assert account not in {r.user_id for r in await cli._records()}


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
