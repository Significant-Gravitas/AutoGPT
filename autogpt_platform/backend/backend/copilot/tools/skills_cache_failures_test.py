"""A partially completed write must not leave the cached skill index stale."""

import asyncio
from unittest.mock import patch

import pytest

from backend.copilot.tools import skills
from backend.copilot.tools.skills_test import _cached_skills, _FakeWorkspaceManager


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("write acknowledgement lost"), asyncio.CancelledError()]
)
async def test_a_stored_package_reaches_the_index_when_the_write_does_not_return(error):
    fake = _FakeWorkspaceManager()
    write = fake.write_skill_package

    async def write_then_fail(**kwargs):
        await write(**kwargs)
        raise error

    with _cached_skills(fake):
        assert await skills.list_user_skills("user-1") == []
        with (
            patch.object(fake, "write_skill_package", write_then_fail),
            pytest.raises(type(error)),
        ):
            await skills.store_user_skill(
                "user-1", name="fresh", description="description", body="steps"
            )

        assert "/skills/fresh/SKILL.md" in fake.files
        assert [skill.name for skill in await skills.list_user_skills("user-1")] == [
            "fresh"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("cleanup failed"), asyncio.CancelledError()]
)
async def test_a_deleted_skill_leaves_the_index_when_cleanup_does_not_finish(error):
    fake = _FakeWorkspaceManager()
    delete = fake.delete_file

    async def delete_then_fail(file_id):
        await delete(file_id)
        raise error

    with _cached_skills(fake):
        await skills.store_user_skill(
            "user-1", name="gone", description="description", body="steps"
        )
        assert [skill.name for skill in await skills.list_user_skills("user-1")] == [
            "gone"
        ]
        with (
            patch.object(fake, "delete_file", delete_then_fail),
            pytest.raises(type(error)),
        ):
            await skills.delete_user_skill("user-1", "gone")

        assert "/skills/gone/SKILL.md" not in fake.files
        assert await skills.list_user_skills("user-1") == []
