from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.blocks.reddit import TEST_CREDENTIALS
from backend.blocks.reddit_moderation import (
    ApproveRedditPostBlock,
    BanSubredditUserBlock,
    LockRedditPostBlock,
    ModQueueBlock,
    RemoveRedditPostBlock,
    SendModMailBlock,
    UnbanSubredditUserBlock,
)


@pytest.mark.parametrize(
    "block_cls",
    [
        ModQueueBlock,
        RemoveRedditPostBlock,
        ApproveRedditPostBlock,
        LockRedditPostBlock,
        BanSubredditUserBlock,
        UnbanSubredditUserBlock,
        SendModMailBlock,
    ],
)
def test_moderation_blocks_have_credentials_field(block_cls):
    block = block_cls()

    field = block.input_schema.model_fields["credentials"]
    assert field is not None


def test_get_mod_queue_uses_modqueue_and_submission_fullnames(mocker):
    queued_item = SimpleNamespace(
        id="abc123",
        fullname="t3_abc123",
        title="Queued title",
        author="queued-user",
        permalink="/r/test/comments/abc123/queued_title/",
        mod_reason_title="",
    )
    sub = MagicMock()
    sub.mod.modqueue.return_value = [queued_item]
    client = MagicMock()
    client.subreddit.return_value = sub
    mocker.patch("backend.blocks.reddit_moderation.get_praw", return_value=client)

    items = ModQueueBlock.get_mod_queue(
        TEST_CREDENTIALS,
        subreddit="test",
        limit=5,
        only="submissions",
    )

    sub.mod.modqueue.assert_called_once_with(limit=5, only="submissions")
    assert items == [
        {
            "id": "t3_abc123",
            "type": "submission",
            "title": "Queued title",
            "author": "queued-user",
            "permalink": "/r/test/comments/abc123/queued_title/",
            "reason": "",
        }
    ]


def test_get_mod_queue_preserves_comment_fullnames(mocker):
    queued_item = SimpleNamespace(
        id="xyz789",
        fullname="t1_xyz789",
        author=None,
        permalink="/r/test/comments/abc123/comment/",
        mod_reason_title=None,
    )
    sub = MagicMock()
    sub.mod.modqueue.return_value = [queued_item]
    client = MagicMock()
    client.subreddit.return_value = sub
    mocker.patch("backend.blocks.reddit_moderation.get_praw", return_value=client)

    items = ModQueueBlock.get_mod_queue(
        TEST_CREDENTIALS,
        subreddit="test",
        limit=5,
        only="comments",
    )

    assert items[0]["id"] == "t1_xyz789"
    assert items[0]["type"] == "comment"
    assert items[0]["title"] == "[comment]"
    assert items[0]["author"] == "[deleted]"


def test_remove_post_accepts_comment_fullname_and_truncates_mod_note(mocker):
    moderated_comment = MagicMock()
    moderated_comment.mod = MagicMock()
    client = MagicMock()
    client.comment.return_value = moderated_comment
    mocker.patch("backend.blocks.reddit_moderation.get_praw", return_value=client)

    result = RemoveRedditPostBlock.remove_post(
        TEST_CREDENTIALS,
        post_id="t1_xyz789",
        spam=False,
        mod_note="x" * 300,
    )

    assert result is True
    client.comment.assert_called_once_with(id="xyz789")
    moderated_comment.mod.remove.assert_called_once_with(
        spam=False,
        mod_note="x" * 250,
    )


def test_ban_user_rejects_non_positive_duration(mocker):
    client = MagicMock()
    subreddit = MagicMock()
    client.subreddit.return_value = subreddit
    mocker.patch("backend.blocks.reddit_moderation.get_praw", return_value=client)

    with pytest.raises(ValueError, match="positive number of days"):
        BanSubredditUserBlock.ban_user(
            TEST_CREDENTIALS,
            subreddit="testsubreddit",
            username="spamuser123",
            duration=0,
            reason="Spam",
            mod_note=None,
            ban_message=None,
        )

    subreddit.banned.add.assert_not_called()
