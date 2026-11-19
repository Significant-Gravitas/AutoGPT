"""Real-database coverage for library folder name uniqueness and nesting depth.

A live folder name is unique among its siblings, roots included, and a
soft-deleted folder gives its name back. Both rules live in partial unique
indexes, so only a real database can show them working.

Folders nest at most ``MAX_FOLDER_DEPTH`` levels deep, and the tree never comes
back deeper than that, whatever the table holds.
"""

import uuid

import prisma.models
import pytest

from backend.api.features.library import db
from backend.api.features.library.exceptions import (
    FolderAlreadyExistsError,
    FolderValidationError,
)
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


@pytest.mark.asyncio(loop_scope="session")
async def test_folders_cannot_be_created_deeper_than_the_cap(server: SpinTestServer):
    user_id = await _user()
    parent_id = None
    for _ in range(db.MAX_FOLDER_DEPTH):
        parent_id = (await db.create_folder(user_id, "Level", parent_id=parent_id)).id

    with pytest.raises(FolderValidationError):
        await db.create_folder(user_id, "Too deep", parent_id=parent_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_folders_cannot_be_moved_deeper_than_the_cap(server: SpinTestServer):
    user_id = await _user()
    chain: list[str] = []
    for _ in range(db.MAX_FOLDER_DEPTH - 1):
        parent_id = chain[-1] if chain else None
        chain.append((await db.create_folder(user_id, "Level", parent_id=parent_id)).id)
    other = await db.create_folder(user_id, "Other")
    child = await db.create_folder(user_id, "Child", parent_id=other.id)

    # The folder itself would fit under the deepest level, but its child would not.
    with pytest.raises(FolderValidationError):
        await db.move_folder(other.id, user_id, target_parent_id=chain[-1])

    moved = await db.move_folder(child.id, user_id, target_parent_id=chain[-1])
    assert moved.parent_id == chain[-1]


@pytest.mark.asyncio(loop_scope="session")
async def test_tree_lists_folders_stored_deeper_than_the_cap_at_the_top(
    server: SpinTestServer,
):
    user_id = await _user()
    depth = db.MAX_FOLDER_DEPTH + 3
    parent_id = None
    for level in range(depth):
        # Straight to the table: rows like these predate the cap.
        row = await prisma.models.LibraryFolder.prisma().create(
            data={"name": f"Level {level}", "userId": user_id, "parentId": parent_id}
        )
        parent_id = row.id

    tree = await db.get_folder_tree(user_id)

    assert len(db.collect_tree_ids(tree)) == depth
    assert len(tree) == 4
    levels, node = 1, tree[0]
    while node.children:
        levels, node = levels + 1, node.children[0]
    assert levels == db.MAX_FOLDER_DEPTH


def test_folder_depths_survive_long_chains_and_cycles():
    chain: dict[str, str | None] = {str(i): str(i - 1) for i in range(1, 5000)}
    chain["0"] = None
    depths = db._folder_depths(chain)
    assert depths["0"] == 1
    assert depths["4999"] == 5000

    # An orphan counts from the top; a cycle ends the walk instead of spinning.
    assert db._folder_depths({"orphan": "gone"}) == {"orphan": 1}
    assert set(db._folder_depths({"a": "b", "b": "a"})) == {"a", "b"}
