"""Real-database coverage for library folder name uniqueness.

A live folder name is unique among its siblings, roots included, and a
soft-deleted folder gives its name back. Both rules live in partial unique
indexes, so only a real database can show them working.
"""

import uuid

import pytest

from backend.api.features.library import db
from backend.api.features.library.exceptions import FolderAlreadyExistsError
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer


async def _user() -> str:
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"library-folders-{suffix}@example.com",
            "name": "Folder Owner",
        }
    )
    return user.id


@pytest.mark.asyncio(loop_scope="session")
async def test_two_root_folders_cannot_share_a_name(server: SpinTestServer):
    user_id = await _user()
    await db.create_folder(user_id, "Work")

    with pytest.raises(FolderAlreadyExistsError):
        await db.create_folder(user_id, "Work")


@pytest.mark.asyncio(loop_scope="session")
async def test_unique_name_helper_suffixes_a_root_clash(server: SpinTestServer):
    user_id = await _user()
    first = await db.create_folder_with_unique_name(user_id, "Imports")
    second = await db.create_folder_with_unique_name(user_id, "Imports")

    assert first.name == "Imports"
    assert second.name == "Imports (2)"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_deleted_folder_gives_its_name_back(server: SpinTestServer):
    user_id = await _user()
    parent = await db.create_folder(user_id, "Clients")

    for parent_id in (None, parent.id):
        gone = await db.create_folder(user_id, "Archive", parent_id=parent_id)
        await db.delete_folder(gone.id, user_id)

        again = await db.create_folder(user_id, "Archive", parent_id=parent_id)
        assert again.id != gone.id


@pytest.mark.asyncio(loop_scope="session")
async def test_renaming_onto_a_deleted_sibling_name_is_allowed(
    server: SpinTestServer,
):
    user_id = await _user()
    gone = await db.create_folder(user_id, "Drafts")
    await db.delete_folder(gone.id, user_id)
    other = await db.create_folder(user_id, "Scratch")

    renamed = await db.update_folder(other.id, user_id, name="Drafts")
    assert renamed.name == "Drafts"

    live = await db.create_folder(user_id, "Notes")
    with pytest.raises(FolderAlreadyExistsError):
        await db.update_folder(live.id, user_id, name="Drafts")
