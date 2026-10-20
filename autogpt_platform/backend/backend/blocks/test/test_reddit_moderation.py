from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from praw.models import Comment, Submission
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from backend.blocks.reddit import (
    REDDIT_BASE_SCOPES,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    RedditCredentialsField,
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
    _get_thing_type,
)


def _patch_praw(mocker) -> MagicMock:
    client = MagicMock()
    mocker.patch("backend.blocks.reddit_moderation.get_praw", return_value=client)
    return client


def _mock_praw_client() -> MagicMock:
    client = MagicMock()
    client.config.kinds = {"comment": "t1", "submission": "t3"}
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


def test_explicit_empty_reddit_scopes_still_include_the_baseline():
    field = RedditCredentialsField(required_scopes=set())
    assert isinstance(field, FieldInfo)
    extra = field.json_schema_extra
    assert isinstance(extra, dict)

    assert set(extra["credentials_scopes"]) == REDDIT_BASE_SCOPES


def test_default_reddit_scopes_remain_implicit_for_legacy_blocks():
    field = RedditCredentialsField()
    assert isinstance(field, FieldInfo)
    extra = field.json_schema_extra
    assert isinstance(extra, dict)

    # The OAuth handler fills in DEFAULT_SCOPES for an absent scope list. Keeping
    # this implicit avoids changing the credential contract of existing blocks.
    assert "credentials_scopes" not in extra


@pytest.mark.parametrize(
    ("queued_item", "only", "expected"),
    [
        pytest.param(
            SimpleNamespace(
                id="abc123",
                fullname="t3_abc123",
                title="Queued title",
                author="queued-user",
                permalink="/r/test/comments/abc123/queued_title/",
                mod_reason_title="",
            ),
            "submissions",
            {
                "id": "t3_abc123",
                "type": "submission",
                "title": "Queued title",
                "author": "queued-user",
                "permalink": "/r/test/comments/abc123/queued_title/",
                "reason": "",
            },
            id="submission",
        ),
        pytest.param(
            # No `title`, no author, no mod reason: the comment defaults.
            SimpleNamespace(
                id="xyz789",
                fullname="t1_xyz789",
                author=None,
                permalink="/r/test/comments/abc123/comment/",
                mod_reason_title=None,
            ),
            "comments",
            {
                "id": "t1_xyz789",
                "type": "comment",
                "title": "[comment]",
                "author": "[deleted]",
                "permalink": "/r/test/comments/abc123/comment/",
                "reason": "",
            },
            id="comment",
        ),
    ],
)
def test_get_mod_queue_maps_listing_items(mocker, queued_item, only, expected):
    sub = MagicMock()
    sub.mod.modqueue.return_value = [queued_item]
    client = _patch_praw(mocker)
    client.subreddit.return_value = sub

    items = ModQueueBlock.get_mod_queue(
        TEST_CREDENTIALS,
        subreddit="test",
        limit=5,
        only=only,
    )

    sub.mod.modqueue.assert_called_once_with(limit=5, only=only)
    assert items == [expected]


@pytest.mark.parametrize(
    ("queued_item", "expected_id", "expected_type"),
    [
        (
            Comment(
                _mock_praw_client(),
                _data={
                    "id": "xyz789",
                    "author": "queued-user",
                    "permalink": "/r/test/comments/abc123/comment/xyz789/",
                    "mod_reason_title": None,
                },
            ),
            "t1_xyz789",
            "comment",
        ),
        (
            Submission(
                _mock_praw_client(),
                _data={
                    "id": "abc123",
                    "author": "queued-user",
                    "permalink": "/r/test/comments/abc123/queued_title/",
                    "mod_reason_title": None,
                    "title": "Queued title",
                },
            ),
            "t3_abc123",
            "submission",
        ),
    ],
)
def test_get_mod_queue_uses_hydrated_praw_fields_without_fetching(
    mocker, queued_item, expected_id, expected_type
):
    """Building an item must not cost an API call per queued thing (no N+1).

    Asserted against the praw client rather than against `_fetch`: praw marks a
    comment built from `_data` as `_fetched=True`, so a patched `_fetch` on the
    comment parameter can never fire and that assertion would be vacuous. Every
    lazy attribute access has to go through `_reddit`, so a `to_item` that starts
    reading a field the modqueue listing doesn't return shows up here — for the
    submission parameter, which praw leaves `_fetched=False`, it really does.
    """
    item_client = queued_item._reddit
    sub = MagicMock()
    sub.mod.modqueue.return_value = [queued_item]
    client = _patch_praw(mocker)
    client.subreddit.return_value = sub

    items = ModQueueBlock.get_mod_queue(
        TEST_CREDENTIALS,
        subreddit="test",
        limit=5,
        only=None,
    )

    assert items[0]["id"] == expected_id
    assert items[0]["type"] == expected_type
    assert item_client.method_calls == []


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


def test_get_thing_type_rejects_unknown_prefixes():
    with pytest.raises(ValueError, match="Ambiguous Reddit thing ID"):
        _get_thing_type("t5_subreddit")


@pytest.mark.asyncio
async def test_mod_queue_run_fans_out_every_item_and_emits_one_batch(mocker):
    items = [
        {
            "id": "t3_first",
            "type": "submission",
            "title": "First",
            "author": "alice",
            "permalink": "/r/test/comments/first/",
            "reason": "",
        },
        {
            "id": "t1_second",
            "type": "comment",
            "title": "[comment]",
            "author": "bob",
            "permalink": "/r/test/comments/first/_/second/",
            "reason": "Rule 1",
        },
    ]
    block = ModQueueBlock()
    mocker.patch.object(block, "get_mod_queue", return_value=items)
    input_data = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "subreddit": "test",
            "limit": 2,
        }
    )

    outputs = [
        output async for output in block.run(input_data, credentials=TEST_CREDENTIALS)
    ]

    # Pin the *complete* scalar sequence, not just the post_id pins: the block
    # description promises scalar outputs fan out once per queued item, and
    # asserting only a subset lets a dropped `yield` pass unnoticed.
    assert outputs == [
        ("post_id", "t3_first"),
        ("item_type", "submission"),
        ("post_title", "First"),
        ("author", "alice"),
        ("permalink", "/r/test/comments/first/"),
        ("reason", ""),
        ("post_id", "t1_second"),
        ("item_type", "comment"),
        ("post_title", "[comment]"),
        ("author", "bob"),
        ("permalink", "/r/test/comments/first/_/second/"),
        ("reason", "Rule 1"),
        ("items", items),
    ]


@pytest.mark.parametrize(
    ("post_id", "kind", "bare_id", "spam", "mod_note", "expected_kwargs"),
    [
        pytest.param(
            "t1_xyz789",
            "comment",
            "xyz789",
            False,
            "Rule 3",
            {"spam": False, "mod_note": "Rule 3"},
            id="comment-with-mod-note",
        ),
        pytest.param(
            "t3_abc123",
            "submission",
            "abc123",
            True,
            None,
            {"spam": True},
            id="submission-without-mod-note",
        ),
    ],
)
def test_remove_post_targets_the_right_thing(
    mocker, post_id, kind, bare_id, spam, mod_note, expected_kwargs
):
    client = _patch_praw(mocker)
    target = MagicMock()
    getattr(client, kind).return_value = target

    result = RemoveRedditPostBlock.remove_post(
        TEST_CREDENTIALS,
        post_id=post_id,
        spam=spam,
        mod_note=mod_note,
    )

    assert result is True
    getattr(client, kind).assert_called_once_with(id=bare_id)
    target.mod.remove.assert_called_once_with(**expected_kwargs)


def test_remove_post_rejects_bare_id_before_calling_reddit(mocker):
    client = _patch_praw(mocker)

    with pytest.raises(ValueError, match="Ambiguous Reddit thing ID"):
        RemoveRedditPostBlock.remove_post(
            TEST_CREDENTIALS, post_id="abc123", spam=False, mod_note=None
        )

    client.submission.assert_not_called()
    client.comment.assert_not_called()


@pytest.mark.parametrize(
    ("post_id", "kind", "bare_id"),
    [
        pytest.param("t3_abc123", "submission", "abc123", id="submission"),
        pytest.param("t1_xyz789", "comment", "xyz789", id="comment"),
    ],
)
def test_approve_post_resolves_target_and_approves(mocker, post_id, kind, bare_id):
    client = _patch_praw(mocker)
    target = MagicMock()
    getattr(client, kind).return_value = target

    assert (
        ApproveRedditPostBlock.approve_post(TEST_CREDENTIALS, post_id=post_id) is True
    )

    getattr(client, kind).assert_called_once_with(id=bare_id)
    target.mod.approve.assert_called_once_with()


@pytest.mark.parametrize(
    ("post_id", "kind", "bare_id", "lock", "called", "not_called"),
    [
        pytest.param(
            "t3_abc123", "submission", "abc123", True, "lock", "unlock", id="lock"
        ),
        pytest.param(
            "t1_xyz789", "comment", "xyz789", False, "unlock", "lock", id="unlock"
        ),
    ],
)
def test_set_lock_calls_the_matching_mod_action(
    mocker, post_id, kind, bare_id, lock, called, not_called
):
    client = _patch_praw(mocker)
    target = MagicMock()
    getattr(client, kind).return_value = target

    assert (
        LockRedditPostBlock.set_lock(TEST_CREDENTIALS, post_id=post_id, lock=lock)
        is lock
    )

    getattr(client, kind).assert_called_once_with(id=bare_id)
    getattr(target.mod, called).assert_called_once_with()
    getattr(target.mod, not_called).assert_not_called()


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
            BanSubredditUserBlock,
            subreddit="test",
            username="u",
            reason="x" * 101,
        )

    with pytest.raises(ValidationError):
        _input(
            SendModMailBlock,
            subreddit="test",
            to_username="someuser",
            subject="x" * 101,
            body="hello",
        )

    with pytest.raises(ValidationError):
        _input(
            SendModMailBlock,
            subreddit="test",
            to_username="someuser",
            subject="hello",
            body="x" * 10001,
        )


# The human-in-the-loop gate for automated moderation. `data/graph.py` reads
# `block.is_irreversible_action` to decide whether a run needs approval, so a block
# silently losing the flag would start banning and removing without a prompt.
# Nothing else in the suite pins it.
@pytest.mark.parametrize(
    ("block_cls", "is_sensitive"),
    [
        (RemoveRedditPostBlock, True),
        (ApproveRedditPostBlock, True),
        (LockRedditPostBlock, True),
        (BanSubredditUserBlock, True),
        (UnbanSubredditUserBlock, True),
        (SendModMailBlock, True),
        # Read-only: the mod queue must NOT demand approval to be listed.
        (ModQueueBlock, False),
    ],
)
def test_state_changing_moderation_blocks_are_gated(block_cls, is_sensitive):
    assert block_cls().is_irreversible_action is is_sensitive


@pytest.mark.asyncio
async def test_ban_run_reports_permanent_when_duration_omitted(mocker):
    """A ban with no duration is permanent, and must say so.

    `test_input` always sets a duration, so the `duration=None` branch of the
    `permanent` output is otherwise unexercised: a moderator issuing a permanent
    ban could silently get `permanent: false`.
    """
    block = BanSubredditUserBlock()
    mocker.patch.object(block, "ban_user", return_value=True)
    input_data = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "subreddit": "testsubreddit",
            "username": "spamuser123",
            "reason": "Spam",
        }
    )

    outputs = [
        output async for output in block.run(input_data, credentials=TEST_CREDENTIALS)
    ]

    assert input_data.duration is None
    assert ("permanent", True) in outputs
    assert ("success", True) in outputs


@pytest.mark.asyncio
async def test_ban_run_reports_temporary_when_duration_given(mocker):
    block = BanSubredditUserBlock()
    mocker.patch.object(block, "ban_user", return_value=True)
    input_data = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "subreddit": "testsubreddit",
            "username": "spamuser123",
            "duration": 7,
            "reason": "Spam",
        }
    )

    outputs = [
        output async for output in block.run(input_data, credentials=TEST_CREDENTIALS)
    ]

    assert ("permanent", False) in outputs


@pytest.mark.asyncio
async def test_modmail_success_is_derived_from_conversation_id(mocker):
    """`success` must mean something: no conversation id, no success."""
    block = SendModMailBlock()
    mocker.patch.object(block, "send_modmail", return_value="")
    input_data = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "subreddit": "testsubreddit",
            "to_username": "someuser",
            "subject": "Warning",
            "body": "Please stop.",
        }
    )

    outputs = [
        output async for output in block.run(input_data, credentials=TEST_CREDENTIALS)
    ]

    assert ("success", False) in outputs


@pytest.mark.asyncio
async def test_mod_queue_run_emits_only_the_empty_batch(mocker):
    """An empty queue yields no scalar pins at all, and exactly one ('items', [])."""
    block = ModQueueBlock()
    mocker.patch.object(block, "get_mod_queue", return_value=[])
    input_data = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "subreddit": "test",
            "limit": 5,
        }
    )

    outputs = [
        output async for output in block.run(input_data, credentials=TEST_CREDENTIALS)
    ]

    assert outputs == [("items", [])]
