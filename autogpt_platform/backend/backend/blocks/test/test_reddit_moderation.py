from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from backend.blocks.reddit import (
    REDDIT_BASE_SCOPES,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
)
from backend.blocks.reddit_moderation import (
    BAN_MAX_DURATION_DAYS,
    MOD_QUEUE_MAX_LIMIT,
    ApproveRedditPostBlock,
    BanSubredditUserBlock,
    LockRedditPostBlock,
    ModQueueBlock,
    RemoveRedditPostBlock,
    SendModMailBlock,
    UnbanSubredditUserBlock,
    _get_moderated_thing,
)


def _patch_praw(mocker) -> MagicMock:
    client = MagicMock()
    mocker.patch("backend.blocks.reddit_moderation.get_praw", return_value=client)
    return client


@pytest.mark.parametrize(
    ("block_cls", "elevated_scope"),
    [
        (ModQueueBlock, "modposts"),
        (RemoveRedditPostBlock, "modposts"),
        (ApproveRedditPostBlock, "modposts"),
        (LockRedditPostBlock, "modposts"),
        (BanSubredditUserBlock, "modcontributors"),
        (UnbanSubredditUserBlock, "modcontributors"),
        (SendModMailBlock, "modmail"),
    ],
)
def test_moderation_blocks_declare_least_privilege_scopes(block_cls, elevated_scope):
    """
    Each moderation block asks for exactly one elevated scope on top of the
    baseline. The baseline must be included because
    ``BaseOAuthHandler.handle_default_scopes`` *replaces* ``DEFAULT_SCOPES`` when a
    non-empty scope list is requested — a block asking for only ``modposts`` would
    receive a token that cannot even call ``client.user.me()``.
    """
    field = block_cls().input_schema.model_fields["credentials"]
    extra = field.json_schema_extra
    assert isinstance(extra, dict)
    declared = set(extra["credentials_scopes"])

    assert declared == REDDIT_BASE_SCOPES | {elevated_scope}
    # Never request scopes no block consumes (e.g. modlog).
    assert "modlog" not in declared


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
    client = _patch_praw(mocker)
    client.subreddit.return_value = sub

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
    client = _patch_praw(mocker)
    client.subreddit.return_value = sub

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


def test_get_mod_queue_returns_empty_list_for_empty_queue(mocker):
    sub = MagicMock()
    sub.mod.modqueue.return_value = []
    client = _patch_praw(mocker)
    client.subreddit.return_value = sub

    assert (
        ModQueueBlock.get_mod_queue(
            TEST_CREDENTIALS, subreddit="test", limit=5, only=None
        )
        == []
    )
    sub.mod.modqueue.assert_called_once_with(limit=5)


def test_get_moderated_thing_resolves_comments_and_submissions(mocker):
    client = _patch_praw(mocker)

    _get_moderated_thing(TEST_CREDENTIALS, "t1_xyz789")
    client.comment.assert_called_once_with(id="xyz789")
    client.submission.assert_not_called()

    client.reset_mock()
    _get_moderated_thing(TEST_CREDENTIALS, "t3_abc123")
    client.submission.assert_called_once_with(id="abc123")
    client.comment.assert_not_called()


@pytest.mark.parametrize("thing_id", ["abc123", "t5_subreddit", "", "T1_xyz789"])
def test_get_moderated_thing_rejects_ambiguous_ids(mocker, thing_id):
    """A bare ID is a valid post ID *and* a valid comment ID — never guess."""
    client = _patch_praw(mocker)

    with pytest.raises(ValueError, match="Ambiguous Reddit thing ID"):
        _get_moderated_thing(TEST_CREDENTIALS, thing_id)

    client.comment.assert_not_called()
    client.submission.assert_not_called()


def test_remove_post_targets_comment_with_mod_note(mocker):
    moderated_comment = MagicMock()
    client = _patch_praw(mocker)
    client.comment.return_value = moderated_comment

    result = RemoveRedditPostBlock.remove_post(
        TEST_CREDENTIALS,
        post_id="t1_xyz789",
        spam=False,
        mod_note="Rule 3",
    )

    assert result is True
    client.comment.assert_called_once_with(id="xyz789")
    moderated_comment.mod.remove.assert_called_once_with(spam=False, mod_note="Rule 3")


def test_remove_post_omits_mod_note_when_unset(mocker):
    moderated_submission = MagicMock()
    client = _patch_praw(mocker)
    client.submission.return_value = moderated_submission

    RemoveRedditPostBlock.remove_post(
        TEST_CREDENTIALS,
        post_id="t3_abc123",
        spam=True,
        mod_note=None,
    )

    client.submission.assert_called_once_with(id="abc123")
    moderated_submission.mod.remove.assert_called_once_with(spam=True)


def test_remove_post_rejects_bare_id_before_calling_reddit(mocker):
    client = _patch_praw(mocker)

    with pytest.raises(ValueError, match="Ambiguous Reddit thing ID"):
        RemoveRedditPostBlock.remove_post(
            TEST_CREDENTIALS, post_id="abc123", spam=False, mod_note=None
        )

    client.submission.assert_not_called()
    client.comment.assert_not_called()


def test_approve_post_calls_mod_approve(mocker):
    moderated_submission = MagicMock()
    client = _patch_praw(mocker)
    client.submission.return_value = moderated_submission

    assert (
        ApproveRedditPostBlock.approve_post(TEST_CREDENTIALS, post_id="t3_abc123")
        is True
    )

    client.submission.assert_called_once_with(id="abc123")
    moderated_submission.mod.approve.assert_called_once_with()


def test_approve_post_resolves_comments(mocker):
    moderated_comment = MagicMock()
    client = _patch_praw(mocker)
    client.comment.return_value = moderated_comment

    ApproveRedditPostBlock.approve_post(TEST_CREDENTIALS, post_id="t1_xyz789")

    client.comment.assert_called_once_with(id="xyz789")
    moderated_comment.mod.approve.assert_called_once_with()


def test_set_lock_locks(mocker):
    moderated_submission = MagicMock()
    client = _patch_praw(mocker)
    client.submission.return_value = moderated_submission

    assert (
        LockRedditPostBlock.set_lock(TEST_CREDENTIALS, post_id="t3_abc123", lock=True)
        is True
    )

    moderated_submission.mod.lock.assert_called_once_with()
    moderated_submission.mod.unlock.assert_not_called()


def test_set_lock_unlocks(mocker):
    moderated_comment = MagicMock()
    client = _patch_praw(mocker)
    client.comment.return_value = moderated_comment

    assert (
        LockRedditPostBlock.set_lock(TEST_CREDENTIALS, post_id="t1_xyz789", lock=False)
        is False
    )

    client.comment.assert_called_once_with(id="xyz789")
    moderated_comment.mod.unlock.assert_called_once_with()
    moderated_comment.mod.lock.assert_not_called()


def test_ban_user_passes_full_kwargs(mocker):
    subreddit = MagicMock()
    client = _patch_praw(mocker)
    client.subreddit.return_value = subreddit

    assert (
        BanSubredditUserBlock.ban_user(
            TEST_CREDENTIALS,
            subreddit="testsubreddit",
            username="spamuser123",
            duration=7,
            reason="Spam",
            mod_note="Third strike",
            ban_message="Please stop spamming.",
        )
        is True
    )

    client.subreddit.assert_called_once_with("testsubreddit")
    subreddit.banned.add.assert_called_once_with(
        "spamuser123",
        ban_reason="Spam",
        duration=7,
        note="Third strike",
        ban_message="Please stop spamming.",
    )


def test_ban_user_permanent_ban_omits_duration(mocker):
    subreddit = MagicMock()
    client = _patch_praw(mocker)
    client.subreddit.return_value = subreddit

    BanSubredditUserBlock.ban_user(
        TEST_CREDENTIALS,
        subreddit="testsubreddit",
        username="spamuser123",
        duration=None,
        reason="Spam",
        mod_note=None,
        ban_message=None,
    )

    subreddit.banned.add.assert_called_once_with("spamuser123", ban_reason="Spam")


def test_unban_user_calls_banned_remove(mocker):
    subreddit = MagicMock()
    client = _patch_praw(mocker)
    client.subreddit.return_value = subreddit

    assert (
        UnbanSubredditUserBlock.unban_user(
            TEST_CREDENTIALS, subreddit="testsubreddit", username="rehabilitateduser"
        )
        is True
    )

    client.subreddit.assert_called_once_with("testsubreddit")
    subreddit.banned.remove.assert_called_once_with("rehabilitateduser")


def test_send_modmail_creates_conversation(mocker):
    subreddit = MagicMock()
    subreddit.modmail.create.return_value = SimpleNamespace(id="conv123")
    client = _patch_praw(mocker)
    client.subreddit.return_value = subreddit

    conversation_id = SendModMailBlock.send_modmail(
        TEST_CREDENTIALS,
        subreddit="testsubreddit",
        to_username="someuser",
        subject="Warning: Spam",
        body="Please stop posting promotional content.",
    )

    assert conversation_id == "conv123"
    client.subreddit.assert_called_once_with("testsubreddit")
    subreddit.modmail.create.assert_called_once_with(
        subject="Warning: Spam",
        body="Please stop posting promotional content.",
        recipient="someuser",
    )


def _input(block_cls, **fields) -> None:
    """Validate a block's Input schema — raises ValidationError on bad values."""
    block_cls.Input.model_validate({"credentials": TEST_CREDENTIALS_INPUT, **fields})


def test_mod_queue_limit_is_bounded():
    _input(ModQueueBlock, subreddit="test", limit=MOD_QUEUE_MAX_LIMIT)

    with pytest.raises(ValidationError):
        _input(ModQueueBlock, subreddit="test", limit=MOD_QUEUE_MAX_LIMIT + 1)


@pytest.mark.parametrize("duration", [0, BAN_MAX_DURATION_DAYS + 1])
def test_ban_duration_is_bounded(duration):
    """Reddit caps temporary bans at 999 days; reject out-of-range up front."""
    _input(BanSubredditUserBlock, subreddit="test", username="u", duration=1)

    with pytest.raises(ValidationError):
        _input(BanSubredditUserBlock, subreddit="test", username="u", duration=duration)


def test_moderator_free_text_inputs_are_length_bounded():
    with pytest.raises(ValidationError):
        _input(RemoveRedditPostBlock, post_id="t3_abc123", mod_note="x" * 251)

    with pytest.raises(ValidationError):
        _input(
            BanSubredditUserBlock,
            subreddit="test",
            username="u",
            ban_message="x" * 1001,
        )

    with pytest.raises(ValidationError):
        _input(
            SendModMailBlock,
            subreddit="test",
            to_username="someuser",
            subject="x" * 101,
            body="hello",
        )
