"""The wizard's pick is read from what it sends, kept as picked, and labelled
as people saw it; the business understanding's copy only counts when it is an
exact option ID."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import onboarding_role
from backend.data.onboarding_role import OnboardingRole


# The root conftest spins a full test server for every test via an autouse
# session fixture. These are unit tests over mocks, so shadow it for this
# module, as backend/data/db_test.py does.
@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


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
