"""The wizard's pick is read from what it sends, kept as picked, and labelled
as people saw it; the business understanding's copy only counts when it is an
exact option ID."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import onboarding_role
from backend.data.onboarding_role import OnboardingRole
from backend.data.onboarding_wizard import OnboardingWizardProgress


# The root conftest spins a full test server for every test via an autouse
# session fixture. These are unit tests over mocks, so shadow it for this
# module, as backend/data/db_test.py does.
@pytest.fixture(scope="session")
def server():
    yield None


@pytest.mark.parametrize(
    "answer, choice, other, label",
    [
        ("Founder/CEO", "Founder/CEO", None, "Founder / CEO"),
        (" Sales/BD ", "Sales/BD", None, "Sales / BD"),
        ("Marketing", "Marketing", None, "Marketing"),
        # Other sends what was typed in its place.
        ("  Dentist  ", "Other", "Dentist", "Other"),
        ("marketing", "Other", "marketing", "Other"),
        ("x" * 99 + " yz", "Other", "x" * 99, "Other"),
        # Passes the route's min_length but never comes from the wizard.
        ("   ", "Other", None, "Other"),
    ],
)
def test_the_answer_is_an_option_or_others_text(answer, choice, other, label):
    role = OnboardingRole.from_answer(answer)
    assert (role.choice, role.other, role.label) == (choice, other, label)


def _draft(**fields) -> OnboardingWizardProgress:
    return OnboardingWizardProgress(
        **{
            "version": 1,
            "currentStep": "painPoints",
            "completedSteps": ["team", "autopilot", "role"],
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
    )


@pytest.mark.parametrize(
    "fields, role",
    [
        ({}, OnboardingRole(choice="Marketing")),
        ({"currentStep": "subscription"}, OnboardingRole(choice="Marketing")),
        (
            {"role": "Other", "otherRole": " Dentist "},
            OnboardingRole(choice="Other", other="Dentist"),
        ),
        # Read like the profile submit: an option's ID typed under Other.
        (
            {"role": "Other", "otherRole": "Engineering"},
            OnboardingRole(choice="Engineering"),
        ),
        # What Other was showing before another pick is not the pick.
        ({"otherRole": "Dentist"}, OnboardingRole(choice="Marketing")),
        ({"currentStep": "role"}, None),
        # Back before the role step, which stays completed.
        ({"currentStep": "autopilot"}, None),
        ({"completedSteps": ["team", "autopilot"]}, None),
        ({"role": ""}, None),
        ({"role": "Other", "otherRole": "  "}, None),
        # The wizard's order can move steps: past the role step is wherever
        # the person went after it, as completedSteps records.
        # With payments on, the access step follows the paywall.
        (
            {
                "currentStep": "access",
                "completedSteps": [
                    "team",
                    "autopilot",
                    "role",
                    "painPoints",
                    "hire",
                    "subscription",
                ],
            },
            OnboardingRole(choice="Marketing"),
        ),
        # An experiment that asks the role first.
        (
            {"currentStep": "team", "completedSteps": ["role"]},
            OnboardingRole(choice="Marketing"),
        ),
        (
            {"currentStep": "team", "completedSteps": ["autopilot", "role", "team"]},
            OnboardingRole(choice="Marketing"),
        ),
        # Back to a step finished before the role step, in that order.
        (
            {
                "currentStep": "autopilot",
                "completedSteps": ["autopilot", "role", "team"],
            },
            None,
        ),
    ],
)
def test_the_draft_gives_the_pick_once_the_role_step_is_left(fields, role):
    assert OnboardingRole.from_wizard(_draft(**fields)) == role


@pytest.mark.parametrize(
    "stored, choice",
    [
        ("Engineering", "Engineering"),
        ("HR/People", "HR/People"),
        # Other's text and an AutoPilot rewrite look the same: no pick.
        ("Dentist", None),
        ("decision maker", None),
        ("Marketing manager", None),
        (" Marketing", None),
        (None, None),
        ("", None),
    ],
)
def test_only_an_exact_option_in_the_understanding_counts(stored, choice):
    role = OnboardingRole.from_understanding(stored)
    assert (role.choice if role else None) == choice


@pytest.mark.asyncio
async def test_the_pick_is_kept_on_the_onboarding_row():
    table = MagicMock(upsert=AsyncMock())
    with patch("prisma.models.UserOnboarding.prisma", return_value=table):
        await onboarding_role.save_onboarding_role(
            "user-1", OnboardingRole(choice="Other", other="Dentist")
        )
    table.upsert.assert_awaited_once_with(
        where={"userId": "user-1"},
        data={
            "create": {"userId": "user-1", "role": "Other", "roleOther": "Dentist"},
            "update": {"role": "Other", "roleOther": "Dentist"},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row, role",
    [
        (
            SimpleNamespace(role="Marketing", roleOther=None),
            OnboardingRole(choice="Marketing"),
        ),
        (
            SimpleNamespace(role="Other", roleOther="Dentist"),
            OnboardingRole(choice="Other", other="Dentist"),
        ),
        (SimpleNamespace(role=None, roleOther=None), None),
        (None, None),
    ],
    ids=["option", "other", "not-picked", "no-row"],
)
async def test_the_kept_pick_is_read_back(row, role):
    table = MagicMock(find_unique=AsyncMock(return_value=row))
    with patch("prisma.models.UserOnboarding.prisma", return_value=table):
        assert await onboarding_role.get_onboarding_role("user-1") == role


def _saved(**fields) -> dict:
    """A draft as the row's Json column returns it."""
    return _draft(**fields).model_dump(mode="json")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row, role",
    [
        (
            SimpleNamespace(
                role="Marketing", roleOther=None, wizardProgress=_saved(role="Sales/BD")
            ),
            OnboardingRole(choice="Marketing"),
        ),
        # A draft saved before SECRT-3036, past the role step: its pick.
        (
            SimpleNamespace(role=None, roleOther=None, wizardProgress=_saved()),
            OnboardingRole(choice="Marketing"),
        ),
        (
            SimpleNamespace(
                role=None, roleOther=None, wizardProgress=_saved(currentStep="role")
            ),
            None,
        ),
        (SimpleNamespace(role=None, roleOther=None, wizardProgress={"v": 2}), None),
        (SimpleNamespace(role=None, roleOther=None, wizardProgress=None), None),
        (None, None),
    ],
    ids=["kept", "drafted", "draft-on-role-step", "broken-draft", "no-draft", "no-row"],
)
async def test_the_answered_pick_falls_back_to_the_draft(row, role):
    table = MagicMock(find_unique=AsyncMock(return_value=row))
    with patch("prisma.models.UserOnboarding.prisma", return_value=table):
        assert await onboarding_role.get_answered_role("user-1") == role
