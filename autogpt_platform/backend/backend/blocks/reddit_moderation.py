from typing import Literal

from praw.models import Comment, Submission

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.reddit import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    RedditCredentials,
    RedditCredentialsField,
    RedditCredentialsInput,
    get_praw,
    settings,
    strip_reddit_prefix,
)
from backend.data.model import SchemaField

REMOVE_MOD_NOTE_MAX_LENGTH = 250
BAN_REASON_MAX_LENGTH = 100
BAN_MOD_NOTE_MAX_LENGTH = 300


def _get_moderated_thing(
    creds: RedditCredentials, thing_id: str
) -> Comment | Submission:
    client = get_praw(creds)
    normalized_id = strip_reddit_prefix(thing_id)
    if thing_id.startswith("t1_"):
        return client.comment(id=normalized_id)
    return client.submission(id=normalized_id)


def _get_thing_id(item: Comment | Submission) -> str:
    fullname = getattr(item, "fullname", None)
    if fullname:
        return fullname
    if isinstance(item, Comment):
        return f"t1_{item.id}"
    return f"t3_{item.id}"


def _get_thing_type(item: Comment | Submission) -> Literal["comment", "submission"]:
    if _get_thing_id(item).startswith("t1_"):
        return "comment"
    return "submission"


class ModQueueBlock(Block):
    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        subreddit: str = SchemaField(
            description="Subreddit name, excluding the /r/ prefix",
        )
        limit: int = SchemaField(
            description="Maximum number of items to fetch from the mod queue",
            default=25,
        )
        only: Literal["submissions", "comments"] | None = SchemaField(
            description="Filter to only submissions or only comments. Leave blank for both.",
            default=None,
        )

    class Output(BlockSchemaOutput):
        post_id: str = SchemaField(
            description="Full Reddit thing ID of a queued item, such as 't3_abc123' or 't1_xyz789'"
        )
        item_type: Literal["comment", "submission"] = SchemaField(
            description="Whether the queued item is a comment or submission"
        )
        post_title: str = SchemaField(description="Title of the queued item")
        author: str = SchemaField(description="Username of the author")
        permalink: str = SchemaField(description="Full Reddit permalink")
        reason: str = SchemaField(description="Mod queue reason (if any)")
        items: list[dict] = SchemaField(description="All queued items as a list")

    def __init__(self):
        super().__init__(
            id="166f3083-51da-4cfc-9f7a-57f47b1ba590",
            description="Fetches the mod queue for a subreddit. Requires moderator access.",
            categories={BlockCategory.SOCIAL},
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            input_schema=ModQueueBlock.Input,
            output_schema=ModQueueBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subreddit": "testsubreddit",
                "limit": 5,
            },
            test_output=[
                ("post_id", "t3_abc123"),
                ("item_type", "submission"),
                ("post_title", "Test queued post"),
                ("author", "testuser"),
                ("permalink", "/r/testsubreddit/comments/abc123/test_queued_post/"),
                ("reason", ""),
                (
                    "items",
                    [
                        {
                            "id": "t3_abc123",
                            "type": "submission",
                            "title": "Test queued post",
                            "author": "testuser",
                            "permalink": "/r/testsubreddit/comments/abc123/test_queued_post/",
                            "reason": "",
                        }
                    ],
                ),
            ],
            test_mock={
                "get_mod_queue": lambda creds, subreddit, limit, only: [
                    {
                        "id": "t3_abc123",
                        "type": "submission",
                        "title": "Test queued post",
                        "author": "testuser",
                        "permalink": "/r/testsubreddit/comments/abc123/test_queued_post/",
                        "reason": "",
                    }
                ]
            },
        )

    @staticmethod
    def get_mod_queue(
        creds: RedditCredentials,
        subreddit: str,
        limit: int,
        only: Literal["submissions", "comments"] | None,
    ) -> list[dict]:
        client = get_praw(creds)
        sub = client.subreddit(subreddit)
        kwargs: dict = {"limit": limit}
        if only:
            kwargs["only"] = only
        items = []
        for item in sub.mod.modqueue(**kwargs):
            items.append(
                {
                    "id": _get_thing_id(item),
                    "type": _get_thing_type(item),
                    "title": getattr(item, "title", "[comment]"),
                    "author": str(item.author) if item.author else "[deleted]",
                    "permalink": item.permalink,
                    "reason": getattr(item, "mod_reason_title", "") or "",
                }
            )
        return items

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        items = self.get_mod_queue(
            credentials,
            subreddit=input_data.subreddit,
            limit=input_data.limit,
            only=input_data.only,
        )
        for item in items:
            yield "post_id", item["id"]
            yield "item_type", item["type"]
            yield "post_title", item["title"]
            yield "author", item["author"]
            yield "permalink", item["permalink"]
            yield "reason", item["reason"]
        yield "items", items


class RemoveRedditPostBlock(Block):
    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        post_id: str = SchemaField(
            description="ID or fullname of the post/comment to remove, such as 't3_abc123', 't1_xyz789', or bare submission ID 'abc123'",
        )
        spam: bool = SchemaField(
            description="Mark as spam (True) or just remove (False). Spam trains the filter.",
            default=False,
        )
        mod_note: str | None = SchemaField(
            description="Optional internal moderator note visible only to mods",
            default=None,
            max_length=REMOVE_MOD_NOTE_MAX_LENGTH,
        )

    class Output(BlockSchemaOutput):
        post_id: str = SchemaField(description="ID of the removed post (pass-through)")
        success: bool = SchemaField(description="Whether the removal succeeded")

    def __init__(self):
        super().__init__(
            id="f75643df-0a1a-4240-aa5b-9b2a1b20dcdd",
            description="Removes a Reddit post or comment as a moderator. Requires 'modposts' scope.",
            categories={BlockCategory.SOCIAL},
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            input_schema=RemoveRedditPostBlock.Input,
            output_schema=RemoveRedditPostBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "post_id": "abc123",
                "spam": False,
            },
            test_output=[
                ("post_id", "abc123"),
                ("success", True),
            ],
            test_mock={"remove_post": lambda creds, post_id, spam, mod_note: True},
            is_sensitive_action=True,
        )

    @staticmethod
    def remove_post(
        creds: RedditCredentials,
        post_id: str,
        spam: bool,
        mod_note: str | None,
    ) -> bool:
        thing = _get_moderated_thing(creds, post_id)
        remove_kwargs: dict[str, bool | str] = {"spam": spam}
        if mod_note:
            remove_kwargs["mod_note"] = mod_note[:REMOVE_MOD_NOTE_MAX_LENGTH]
        thing.mod.remove(**remove_kwargs)
        return True

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        success = self.remove_post(
            credentials,
            post_id=input_data.post_id,
            spam=input_data.spam,
            mod_note=input_data.mod_note,
        )
        yield "post_id", input_data.post_id
        yield "success", success


class ApproveRedditPostBlock(Block):
    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        post_id: str = SchemaField(
            description="ID or fullname of the post/comment to approve, such as 't3_abc123', 't1_xyz789', or bare submission ID 'abc123'",
        )

    class Output(BlockSchemaOutput):
        post_id: str = SchemaField(description="ID of the approved post (pass-through)")
        success: bool = SchemaField(description="Whether the approval succeeded")

    def __init__(self):
        super().__init__(
            id="ae695fcf-e1bf-4900-b06c-3ae21d6edf70",
            description="Approves a Reddit post or comment from the mod queue. Requires 'modposts' scope.",
            categories={BlockCategory.SOCIAL},
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            input_schema=ApproveRedditPostBlock.Input,
            output_schema=ApproveRedditPostBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "post_id": "abc123",
            },
            test_output=[
                ("post_id", "abc123"),
                ("success", True),
            ],
            test_mock={"approve_post": lambda creds, post_id: True},
            is_sensitive_action=True,
        )

    @staticmethod
    def approve_post(creds: RedditCredentials, post_id: str) -> bool:
        thing = _get_moderated_thing(creds, post_id)
        thing.mod.approve()
        return True

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        success = self.approve_post(credentials, post_id=input_data.post_id)
        yield "post_id", input_data.post_id
        yield "success", success


class LockRedditPostBlock(Block):
    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        post_id: str = SchemaField(
            description="ID or fullname of the post/comment to lock or unlock",
        )
        lock: bool = SchemaField(
            description="True to lock (disable comments/replies), False to unlock",
            default=True,
        )

    class Output(BlockSchemaOutput):
        post_id: str = SchemaField(description="ID of the post (pass-through)")
        locked: bool = SchemaField(description="Current lock state after the action")

    def __init__(self):
        super().__init__(
            id="1deaf67c-0407-457f-989d-323198073f74",
            description="Locks or unlocks a Reddit post or comment to prevent or allow replies. Requires 'modposts' scope.",
            categories={BlockCategory.SOCIAL},
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            input_schema=LockRedditPostBlock.Input,
            output_schema=LockRedditPostBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "post_id": "abc123",
                "lock": True,
            },
            test_output=[
                ("post_id", "abc123"),
                ("locked", True),
            ],
            test_mock={"set_lock": lambda creds, post_id, lock: lock},
            is_sensitive_action=True,
        )

    @staticmethod
    def set_lock(creds: RedditCredentials, post_id: str, lock: bool) -> bool:
        thing = _get_moderated_thing(creds, post_id)
        if lock:
            thing.mod.lock()
        else:
            thing.mod.unlock()
        return lock

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        locked = self.set_lock(
            credentials,
            post_id=input_data.post_id,
            lock=input_data.lock,
        )
        yield "post_id", input_data.post_id
        yield "locked", locked


class BanSubredditUserBlock(Block):
    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        subreddit: str = SchemaField(
            description="Subreddit to ban the user from, excluding the /r/ prefix",
        )
        username: str = SchemaField(
            description="Reddit username to ban (without the u/ prefix)",
        )
        duration: int | None = SchemaField(
            description="Ban duration in days. Leave blank for a permanent ban.",
            default=None,
            ge=1,
        )
        reason: str = SchemaField(
            description="Internal moderator-only ban reason (max 100 chars). Use ban_message to explain the ban to the user.",
            default="Violation of subreddit rules",
            max_length=BAN_REASON_MAX_LENGTH,
        )
        mod_note: str | None = SchemaField(
            description="Internal moderator note (not shown to the user)",
            default=None,
            max_length=BAN_MOD_NOTE_MAX_LENGTH,
        )
        ban_message: str | None = SchemaField(
            description="Optional custom message sent to the user explaining the ban",
            default=None,
        )

    class Output(BlockSchemaOutput):
        username: str = SchemaField(description="Banned username (pass-through)")
        subreddit: str = SchemaField(description="Subreddit (pass-through)")
        success: bool = SchemaField(description="Whether the ban was applied")
        permanent: bool = SchemaField(description="True if the ban is permanent")

    def __init__(self):
        super().__init__(
            id="428d56d4-52d0-47d9-8544-836d13d196c0",
            description="Bans a user from a subreddit. Requires 'modcontributors' scope.",
            categories={BlockCategory.SOCIAL},
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            input_schema=BanSubredditUserBlock.Input,
            output_schema=BanSubredditUserBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subreddit": "testsubreddit",
                "username": "spamuser123",
                "duration": 7,
                "reason": "Spam",
            },
            test_output=[
                ("username", "spamuser123"),
                ("subreddit", "testsubreddit"),
                ("success", True),
                ("permanent", False),
            ],
            test_mock={
                "ban_user": lambda creds, subreddit, username, duration, reason, mod_note, ban_message: True
            },
            is_sensitive_action=True,
        )

    @staticmethod
    def ban_user(
        creds: RedditCredentials,
        subreddit: str,
        username: str,
        duration: int | None,
        reason: str,
        mod_note: str | None,
        ban_message: str | None,
    ) -> bool:
        if duration is not None and duration <= 0:
            raise ValueError("Ban duration must be a positive number of days.")

        client = get_praw(creds)
        sub = client.subreddit(subreddit)
        ban_kwargs: dict = {"ban_reason": reason[:BAN_REASON_MAX_LENGTH]}
        if duration is not None:
            ban_kwargs["duration"] = duration
        if mod_note:
            ban_kwargs["note"] = mod_note[:BAN_MOD_NOTE_MAX_LENGTH]
        if ban_message:
            ban_kwargs["ban_message"] = ban_message
        sub.banned.add(username, **ban_kwargs)
        return True

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        success = self.ban_user(
            credentials,
            subreddit=input_data.subreddit,
            username=input_data.username,
            duration=input_data.duration,
            reason=input_data.reason,
            mod_note=input_data.mod_note,
            ban_message=input_data.ban_message,
        )
        yield "username", input_data.username
        yield "subreddit", input_data.subreddit
        yield "success", success
        yield "permanent", input_data.duration is None


class UnbanSubredditUserBlock(Block):
    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        subreddit: str = SchemaField(
            description="Subreddit to unban the user from, excluding the /r/ prefix",
        )
        username: str = SchemaField(
            description="Reddit username to unban (without the u/ prefix)",
        )

    class Output(BlockSchemaOutput):
        username: str = SchemaField(description="Unbanned username (pass-through)")
        subreddit: str = SchemaField(description="Subreddit (pass-through)")
        success: bool = SchemaField(description="Whether the unban succeeded")

    def __init__(self):
        super().__init__(
            id="90979f47-605e-4478-a417-39da3d7184ef",
            description="Unbans a user from a subreddit. Requires 'modcontributors' scope.",
            categories={BlockCategory.SOCIAL},
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            input_schema=UnbanSubredditUserBlock.Input,
            output_schema=UnbanSubredditUserBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subreddit": "testsubreddit",
                "username": "rehabilitateduser",
            },
            test_output=[
                ("username", "rehabilitateduser"),
                ("subreddit", "testsubreddit"),
                ("success", True),
            ],
            test_mock={"unban_user": lambda creds, subreddit, username: True},
            is_sensitive_action=True,
        )

    @staticmethod
    def unban_user(creds: RedditCredentials, subreddit: str, username: str) -> bool:
        client = get_praw(creds)
        sub = client.subreddit(subreddit)
        sub.banned.remove(username)
        return True

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        success = self.unban_user(
            credentials,
            subreddit=input_data.subreddit,
            username=input_data.username,
        )
        yield "username", input_data.username
        yield "subreddit", input_data.subreddit
        yield "success", success


class SendModMailBlock(Block):
    class Input(BlockSchemaInput):
        credentials: RedditCredentialsInput = RedditCredentialsField()
        subreddit: str = SchemaField(
            description="Subreddit to send modmail from, excluding the /r/ prefix",
        )
        to_username: str = SchemaField(
            description="Username to send the modmail to (without u/ prefix)",
        )
        subject: str = SchemaField(
            description="Subject line of the modmail message",
        )
        body: str = SchemaField(
            description="Body of the modmail message",
        )

    class Output(BlockSchemaOutput):
        conversation_id: str = SchemaField(
            description="ID of the created modmail conversation"
        )
        success: bool = SchemaField(description="Whether the modmail was sent")

    def __init__(self):
        super().__init__(
            id="168b919c-0e06-471d-bd46-eb354ed3d278",
            description="Sends a modmail message from a subreddit to a user. Requires 'modmail' scope.",
            categories={BlockCategory.SOCIAL},
            disabled=(
                not settings.secrets.reddit_client_id
                or not settings.secrets.reddit_client_secret
            ),
            input_schema=SendModMailBlock.Input,
            output_schema=SendModMailBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subreddit": "testsubreddit",
                "to_username": "someuser",
                "subject": "Warning: Spam",
                "body": "Please stop posting promotional content.",
            },
            test_output=[
                ("conversation_id", "mock_conv_id"),
                ("success", True),
            ],
            test_mock={
                "send_modmail": lambda creds, subreddit, to_username, subject, body: "mock_conv_id"
            },
            is_sensitive_action=True,
        )

    @staticmethod
    def send_modmail(
        creds: RedditCredentials,
        subreddit: str,
        to_username: str,
        subject: str,
        body: str,
    ) -> str:
        client = get_praw(creds)
        sub = client.subreddit(subreddit)
        conversation = sub.modmail.create(
            subject=subject,
            body=body,
            recipient=to_username,
        )
        return conversation.id

    async def run(
        self, input_data: Input, *, credentials: RedditCredentials, **kwargs
    ) -> BlockOutput:
        conversation_id = self.send_modmail(
            credentials,
            subreddit=input_data.subreddit,
            to_username=input_data.to_username,
            subject=input_data.subject,
            body=input_data.body,
        )
        yield "conversation_id", conversation_id
        yield "success", True
