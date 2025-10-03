from typing import Type

import pytest

from backend.data.block import Block, get_blocks
from backend.util.test import execute_block_test


@pytest.mark.parametrize("block", get_blocks().values(), ids=lambda b: b.name)
async def test_available_blocks(block: Type[Block]):
    await execute_block_test(block())


@pytest.mark.parametrize("block", get_blocks().values(), ids=lambda b: b.name)
async def test_block_ids_valid(block: Type[Block]):
    # add the tests here to check they are uuid4
    import uuid

    # Skip list for blocks with known invalid UUIDs
    skip_blocks = {
        "GetWeatherInformationBlock",
        "ExecuteCodeBlock",
        "CountdownTimerBlock",
        "TwitterGetListTweetsBlock",
        "TwitterRemoveListMemberBlock",
        "TwitterAddListMemberBlock",
        "TwitterGetListMembersBlock",
        "TwitterGetListMembershipsBlock",
        "TwitterUnfollowListBlock",
        "TwitterFollowListBlock",
        "TwitterUnpinListBlock",
        "TwitterPinListBlock",
        "TwitterGetPinnedListsBlock",
        "TwitterDeleteListBlock",
        "TwitterUpdateListBlock",
        "TwitterCreateListBlock",
        "TwitterGetListBlock",
        "TwitterGetOwnedListsBlock",
        "TwitterGetSpacesBlock",
        "TwitterGetSpaceByIdBlock",
        "TwitterGetSpaceBuyersBlock",
        "TwitterGetSpaceTweetsBlock",
        "TwitterSearchSpacesBlock",
        "TwitterGetUserMentionsBlock",
        "TwitterGetHomeTimelineBlock",
        "TwitterGetUserTweetsBlock",
        "TwitterGetTweetBlock",
        "TwitterGetTweetsBlock",
        "TwitterGetQuoteTweetsBlock",
        "TwitterLikeTweetBlock",
        "TwitterGetLikingUsersBlock",
        "TwitterGetLikedTweetsBlock",
        "TwitterUnlikeTweetBlock",
        "TwitterBookmarkTweetBlock",
        "TwitterGetBookmarkedTweetsBlock",
        "TwitterRemoveBookmarkTweetBlock",
        "TwitterRetweetBlock",
        "TwitterRemoveRetweetBlock",
        "TwitterGetRetweetersBlock",
        "TwitterHideReplyBlock",
        "TwitterUnhideReplyBlock",
        "TwitterPostTweetBlock",
        "TwitterDeleteTweetBlock",
        "TwitterSearchRecentTweetsBlock",
        "TwitterUnfollowUserBlock",
        "TwitterFollowUserBlock",
        "TwitterGetFollowersBlock",
        "TwitterGetFollowingBlock",
        "TwitterUnmuteUserBlock",
        "TwitterGetMutedUsersBlock",
        "TwitterMuteUserBlock",
        "TwitterGetBlockedUsersBlock",
        "TwitterGetUserBlock",
        "TwitterGetUsersBlock",
        "TodoistCreateLabelBlock",
        "TodoistListLabelsBlock",
        "TodoistGetLabelBlock",
        "TodoistUpdateLabelBlock",
        "TodoistDeleteLabelBlock",
        "TodoistGetSharedLabelsBlock",
        "TodoistRenameSharedLabelsBlock",
        "TodoistRemoveSharedLabelsBlock",
        "TodoistCreateTaskBlock",
        "TodoistGetTasksBlock",
        "TodoistGetTaskBlock",
        "TodoistUpdateTaskBlock",
        "TodoistCloseTaskBlock",
        "TodoistReopenTaskBlock",
        "TodoistDeleteTaskBlock",
        "TodoistListSectionsBlock",
        "TodoistGetSectionBlock",
        "TodoistDeleteSectionBlock",
        "TodoistCreateProjectBlock",
        "TodoistGetProjectBlock",
        "TodoistUpdateProjectBlock",
        "TodoistDeleteProjectBlock",
        "TodoistListCollaboratorsBlock",
        "TodoistGetCommentsBlock",
        "TodoistGetCommentBlock",
        "TodoistUpdateCommentBlock",
        "TodoistDeleteCommentBlock",
        "GithubListStargazersBlock",
        "Slant3DSlicerBlock",
    }

    block_instance = block()

    # Skip blocks with known invalid UUIDs
    if block_instance.__class__.__name__ in skip_blocks:
        pytest.skip(
            f"Skipping UUID check for {block_instance.__class__.__name__} - known invalid UUID"
        )

    # Check that the ID is not empty
    assert block_instance.id, f"Block {block.name} has empty ID"

    # Check that the ID is a valid UUID4
    try:
        parsed_uuid = uuid.UUID(block_instance.id)
        # Verify it's specifically UUID version 4
        assert (
            parsed_uuid.version == 4
        ), f"Block {block.name} ID is UUID version {parsed_uuid.version}, expected version 4"
    except ValueError:
        pytest.fail(f"Block {block.name} has invalid UUID format: {block_instance.id}")
