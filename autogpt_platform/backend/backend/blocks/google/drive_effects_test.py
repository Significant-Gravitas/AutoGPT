import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.drive_comments import GoogleDriveListCommentsBlock
from backend.blocks.google.drive_files import (
    GoogleDriveDownloadFileBlock,
    GoogleDriveGetFileInfoBlock,
    GoogleDriveGetFilePermissionsBlock,
    GoogleDriveReadFileBlock,
)
from backend.blocks.google.drive_manage import (
    GoogleDriveCopyFileBlock,
    GoogleDriveCreateFileBlock,
    GoogleDriveCreateFolderBlock,
    GoogleDriveMoveFileBlock,
)
from backend.blocks.google.drive_search import (
    GoogleDriveListRecentFilesBlock,
    GoogleDriveSearchFilesBlock,
)


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GoogleDriveSearchFilesBlock, BlockEffect.READ),
        (GoogleDriveListRecentFilesBlock, BlockEffect.READ),
        (GoogleDriveGetFileInfoBlock, BlockEffect.READ),
        (GoogleDriveReadFileBlock, BlockEffect.READ),
        (GoogleDriveDownloadFileBlock, BlockEffect.WORKSPACE),
        (GoogleDriveGetFilePermissionsBlock, BlockEffect.READ),
        (GoogleDriveCreateFileBlock, BlockEffect.EXTERNAL),
        (GoogleDriveCreateFolderBlock, BlockEffect.EXTERNAL),
        (GoogleDriveCopyFileBlock, BlockEffect.EXTERNAL),
        (GoogleDriveMoveFileBlock, BlockEffect.EXTERNAL),
        (GoogleDriveListCommentsBlock, BlockEffect.READ),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
