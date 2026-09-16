"""Soft-deleting a workspace file frees its path for the next write.

``write_file(overwrite=True)`` soft-deletes the row it replaces, so the renamed
path is what two writes to one path race on.
"""

from datetime import datetime, timedelta, timezone

import pytest

from backend.data import workspace

WITHIN_ONE_SECOND = [
    datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc) + timedelta(milliseconds=ms)
    for ms in (0, 120)
]


class _Row:
    def __init__(self, path: str):
        self.id = "file-1"
        self.path = path


@pytest.fixture
def renames(mocker):
    """Every path ``soft_delete_workspace_file`` renames a row to, with a clock
    that advances inside one second so only the suffix separates two calls."""
    mocker.patch.object(
        workspace, "get_workspace_file", mocker.AsyncMock(return_value=_Row("/a/b.txt"))
    )
    mocker.patch.object(
        workspace,
        "datetime",
        mocker.MagicMock(now=mocker.MagicMock(side_effect=WITHIN_ONE_SECOND)),
    )
    captured: list[str] = []
    prisma = mocker.MagicMock()
    prisma.update = mocker.AsyncMock(
        side_effect=lambda where, data: captured.append(data["path"])
    )
    mocker.patch.object(
        workspace.UserWorkspaceFile, "prisma", mocker.MagicMock(return_value=prisma)
    )
    return captured


async def test_two_deletes_in_one_second_free_the_path_twice(renames: list[str]):
    """A second-resolution suffix gave both the same name, so the second rename
    failed the unique index it exists to get out of the way of — leaving a file
    the user could no longer overwrite."""
    await workspace.soft_delete_workspace_file("file-1", "ws-1")
    await workspace.soft_delete_workspace_file("file-1", "ws-1")

    assert renames[0] != renames[1]
    assert all(name.startswith("/a/b.txt__deleted__") for name in renames)
