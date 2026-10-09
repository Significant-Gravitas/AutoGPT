"""Reassignments must not retain an earlier database alias."""

import pytest

from backend.util.db_boundary import find_violations

USER_SOURCE = """
from prisma.models import User

async def read_user(user_id):
    return await User.prisma().find_unique(where={"id": user_id})
"""


@pytest.mark.parametrize(
    "assignment",
    [
        "client: object = payload",
        "client, other = payload",
        "[client, other] = payload",
        "other, (client, tail) = payload",
        "other, *client = payload",
        "other, *[client, tail] = payload",
    ],
)
def test_unresolved_reassignment_clears_database_aliases(assignment: str):
    sources = {
        "backend.data.user": USER_SOURCE,
        "backend.notifications.example": (
            "from backend.data import user\n"
            "async def notify(payload):\n"
            "    client = user\n"
            f"    {assignment}\n"
            "    return await client.read_user('user')\n"
        ),
    }
    assert not find_violations(sources)


@pytest.mark.parametrize(
    "assignment",
    ["reader = payload", "reader: object = payload", "reader, other = payload"],
)
def test_reassigned_exports_do_not_taint_consumers(assignment: str):
    sources = {
        "backend.data.user": USER_SOURCE,
        "backend.util.lookup": (
            "from backend.data.user import read_user\n"
            "reader = read_user\n"
            f"{assignment}\n"
        ),
        "backend.notifications.example": (
            "from backend.util.lookup import reader\n"
            "async def notify():\n    return await reader('user')\n"
        ),
    }
    assert not any(
        key.startswith("notifications/example.py::") for key in find_violations(sources)
    )


@pytest.mark.parametrize(
    "assignment", ["client = user", "client: object = user", "client: object"]
)
def test_resolved_reassignments_still_reject_database_access(assignment: str):
    sources = {
        "backend.data.user": USER_SOURCE,
        "backend.notifications.example": (
            "from backend.data import user\n"
            "async def notify(payload):\n"
            "    client = user\n"
            f"    {assignment}\n"
            "    return await client.read_user('user')\n"
        ),
    }
    assert "notifications/example.py::notify -> backend.data.user.read_user" in (
        find_violations(sources)
    )


def test_unpacking_still_checks_database_access_on_the_right_hand_side():
    sources = {
        "backend.data.user": USER_SOURCE,
        "backend.notifications.example": (
            "from backend.data import user\n"
            "async def notify():\n"
            "    client, other = await user.read_user('user')\n"
        ),
    }
    assert "notifications/example.py::notify -> backend.data.user.read_user" in (
        find_violations(sources)
    )
