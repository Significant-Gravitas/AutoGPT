import logging

from typing_extensions import TypedDict

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from ._api import GITHUB_API_URL, get_api
from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)

logger = logging.getLogger(__name__)


class NotificationItem(TypedDict):
    thread_id: str
    reason: str
    unread: bool
    updated_at: str
    title: str
    subject_type: str
    subject_url: str
    repository: str


TEST_NOTIFICATION_ITEM: NotificationItem = {
    "thread_id": "1337",
    "reason": "review_requested",
    "unread": True,
    "updated_at": "2026-07-21T12:00:00Z",
    "title": "Fix the flux capacitor",
    "subject_type": "PullRequest",
    "subject_url": "https://github.com/owner/repo/pull/1",
    "repository": "owner/repo",
}


class GithubListNotificationsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("notifications")
        include_read: bool = SchemaField(
            description="Whether to include notifications that are already "
            "marked as read",
            default=False,
        )
        participating_only: bool = SchemaField(
            description="Whether to only include notifications in which you are "
            "directly participating or mentioned",
            default=False,
        )
        repo: str = SchemaField(
            description="Repository to list notifications for. "
            "Leave empty to list notifications for all repositories.",
            placeholder="{owner}/{repo}",
            default="",
            advanced=True,
        )
        since: str = SchemaField(
            description="Only show notifications updated after the given "
            "ISO 8601 timestamp",
            placeholder="2026-01-01T00:00:00Z",
            default="",
            advanced=True,
        )
        before: str = SchemaField(
            description="Only show notifications updated before the given "
            "ISO 8601 timestamp",
            placeholder="2026-01-01T00:00:00Z",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        notification: NotificationItem = SchemaField(
            title="Notification", description="Each notification thread"
        )
        notifications: list[NotificationItem] = SchemaField(
            description="List of notification threads"
        )
        error: str = SchemaField(
            description="Error message if listing notifications failed"
        )

    def __init__(self):
        super().__init__(
            id="15911223-52e9-47ec-ab95-0581f36c8092",
            description="This block lists GitHub notifications for the authenticated "
            "user, e.g. mentions, review requests, and updates on subscribed threads.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubListNotificationsBlock.Input,
            output_schema=GithubListNotificationsBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("notifications", [TEST_NOTIFICATION_ITEM]),
                ("notification", TEST_NOTIFICATION_ITEM),
            ],
            test_mock={
                "list_notifications": lambda *args, **kwargs: [TEST_NOTIFICATION_ITEM]
            },
        )

    @staticmethod
    async def list_notifications(
        credentials: GithubCredentials,
        include_read: bool,
        participating_only: bool,
        repo: str,
        since: str,
        before: str,
    ) -> list[NotificationItem]:
        api = get_api(credentials, convert_urls=False)
        url = (
            f"{GITHUB_API_URL}/repos/{repo}/notifications"
            if repo
            else f"{GITHUB_API_URL}/notifications"
        )
        params: dict[str, str] = {}
        if include_read:
            params["all"] = "true"
        if participating_only:
            params["participating"] = "true"
        if since:
            params["since"] = since
        if before:
            params["before"] = before

        response = await api.get(url, params=params)
        return [_to_notification_item(thread) for thread in response.json()]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        notifications = await self.list_notifications(
            credentials,
            input_data.include_read,
            input_data.participating_only,
            input_data.repo,
            input_data.since,
            input_data.before,
        )
        yield "notifications", notifications
        for notification in notifications:
            yield "notification", notification


class GithubGetNotificationThreadBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("notifications")
        thread_id: str = SchemaField(
            description="ID of the notification thread",
            placeholder="1337",
        )

    class Output(BlockSchemaOutput):
        notification: NotificationItem = SchemaField(
            description="The notification thread"
        )
        title: str = SchemaField(description="Title of the notification subject")
        reason: str = SchemaField(
            description="Reason you received the notification (e.g. 'mention', "
            "'review_requested', 'subscribed')"
        )
        unread: bool = SchemaField(description="Whether the notification is unread")
        subject_type: str = SchemaField(
            description="Type of the notification subject "
            "(e.g. 'Issue', 'PullRequest', 'Release')"
        )
        subject_url: str = SchemaField(
            description="URL of the notification subject on GitHub"
        )
        repository: str = SchemaField(
            description="Full name of the repository (owner/repo)"
        )
        error: str = SchemaField(
            description="Error message if fetching the notification thread failed"
        )

    def __init__(self):
        super().__init__(
            id="454afb97-46c4-43ca-b54a-543d647a9113",
            description="This block fetches a single GitHub notification thread.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubGetNotificationThreadBlock.Input,
            output_schema=GithubGetNotificationThreadBlock.Output,
            test_input={
                "thread_id": "1337",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("notification", TEST_NOTIFICATION_ITEM),
                ("title", TEST_NOTIFICATION_ITEM["title"]),
                ("reason", TEST_NOTIFICATION_ITEM["reason"]),
                ("unread", TEST_NOTIFICATION_ITEM["unread"]),
                ("subject_type", TEST_NOTIFICATION_ITEM["subject_type"]),
                ("subject_url", TEST_NOTIFICATION_ITEM["subject_url"]),
                ("repository", TEST_NOTIFICATION_ITEM["repository"]),
            ],
            test_mock={
                "get_thread": lambda *args, **kwargs: TEST_NOTIFICATION_ITEM,
            },
        )

    @staticmethod
    async def get_thread(
        credentials: GithubCredentials, thread_id: str
    ) -> NotificationItem:
        api = get_api(credentials, convert_urls=False)
        response = await api.get(f"{GITHUB_API_URL}/notifications/threads/{thread_id}")
        return _to_notification_item(response.json())

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        notification = await self.get_thread(credentials, input_data.thread_id)
        yield "notification", notification
        yield "title", notification["title"]
        yield "reason", notification["reason"]
        yield "unread", notification["unread"]
        yield "subject_type", notification["subject_type"]
        yield "subject_url", notification["subject_url"]
        yield "repository", notification["repository"]


class GithubMarkNotificationsAsReadBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("notifications")
        repo: str = SchemaField(
            description="Repository to mark notifications as read for. "
            "Leave empty to mark notifications for all repositories.",
            placeholder="{owner}/{repo}",
            default="",
            advanced=False,
        )
        last_read_at: str = SchemaField(
            description="Only mark notifications updated before the given "
            "ISO 8601 timestamp as read. Defaults to the current time.",
            placeholder="2026-01-01T00:00:00Z",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="Whether the notifications were marked as read"
        )
        error: str = SchemaField(
            description="Error message if marking notifications as read failed"
        )

    def __init__(self):
        super().__init__(
            id="725d4f50-f40e-4dc9-9487-17bf16d34111",
            description="This block marks all GitHub notifications as read, "
            "optionally scoped to a single repository.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubMarkNotificationsAsReadBlock.Input,
            output_schema=GithubMarkNotificationsAsReadBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"mark_all_as_read": lambda *args, **kwargs: True},
        )

    @staticmethod
    async def mark_all_as_read(
        credentials: GithubCredentials, repo: str, last_read_at: str
    ) -> bool:
        api = get_api(credentials, convert_urls=False)
        url = (
            f"{GITHUB_API_URL}/repos/{repo}/notifications"
            if repo
            else f"{GITHUB_API_URL}/notifications"
        )
        data: dict[str, str] = {}
        if last_read_at:
            data["last_read_at"] = last_read_at

        await api.put(url, json=data)
        return True

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        yield "success", await self.mark_all_as_read(
            credentials, input_data.repo, input_data.last_read_at
        )


class GithubMarkNotificationThreadAsReadBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("notifications")
        thread_id: str = SchemaField(
            description="ID of the notification thread",
            placeholder="1337",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="Whether the notification thread was marked as read"
        )
        error: str = SchemaField(
            description="Error message if marking the thread as read failed"
        )

    def __init__(self):
        super().__init__(
            id="530d5158-34f7-4ea1-a91b-134f5eabc3b8",
            description="This block marks a single GitHub notification thread as read.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubMarkNotificationThreadAsReadBlock.Input,
            output_schema=GithubMarkNotificationThreadAsReadBlock.Output,
            test_input={
                "thread_id": "1337",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"mark_thread_as_read": lambda *args, **kwargs: True},
        )

    @staticmethod
    async def mark_thread_as_read(
        credentials: GithubCredentials, thread_id: str
    ) -> bool:
        api = get_api(credentials, convert_urls=False)
        await api.patch(f"{GITHUB_API_URL}/notifications/threads/{thread_id}")
        return True

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        yield "success", await self.mark_thread_as_read(
            credentials, input_data.thread_id
        )


class GithubMarkNotificationThreadAsDoneBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("notifications")
        thread_id: str = SchemaField(
            description="ID of the notification thread",
            placeholder="1337",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="Whether the notification thread was marked as done"
        )
        error: str = SchemaField(
            description="Error message if marking the thread as done failed"
        )

    def __init__(self):
        super().__init__(
            id="6f579efa-df74-4d0c-bda8-d63c8bc37416",
            description="This block marks a GitHub notification thread as done, "
            "removing it from the notification inbox.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubMarkNotificationThreadAsDoneBlock.Input,
            output_schema=GithubMarkNotificationThreadAsDoneBlock.Output,
            test_input={
                "thread_id": "1337",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"mark_thread_as_done": lambda *args, **kwargs: True},
        )

    @staticmethod
    async def mark_thread_as_done(
        credentials: GithubCredentials, thread_id: str
    ) -> bool:
        api = get_api(credentials, convert_urls=False)
        await api.delete(f"{GITHUB_API_URL}/notifications/threads/{thread_id}")
        return True

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        yield "success", await self.mark_thread_as_done(
            credentials, input_data.thread_id
        )


class GithubUnsubscribeNotificationThreadBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("notifications")
        thread_id: str = SchemaField(
            description="ID of the notification thread",
            placeholder="1337",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="Whether you were unsubscribed from the notification thread"
        )
        error: str = SchemaField(
            description="Error message if unsubscribing from the thread failed"
        )

    def __init__(self):
        super().__init__(
            id="0a57b1b4-87c1-4758-91fc-badedcf338dd",
            description="This block unsubscribes you from a GitHub notification "
            "thread, muting future notifications unless you are mentioned again.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=GithubUnsubscribeNotificationThreadBlock.Input,
            output_schema=GithubUnsubscribeNotificationThreadBlock.Output,
            test_input={
                "thread_id": "1337",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"unsubscribe_thread": lambda *args, **kwargs: True},
        )

    @staticmethod
    async def unsubscribe_thread(
        credentials: GithubCredentials, thread_id: str
    ) -> bool:
        api = get_api(credentials, convert_urls=False)
        await api.delete(
            f"{GITHUB_API_URL}/notifications/threads/{thread_id}/subscription"
        )
        return True

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GithubCredentials,
        **kwargs,
    ) -> BlockOutput:
        yield "success", await self.unsubscribe_thread(
            credentials, input_data.thread_id
        )


def _to_notification_item(thread: dict) -> NotificationItem:
    subject = thread.get("subject") or {}
    return {
        "thread_id": thread["id"],
        "reason": thread["reason"],
        "unread": thread["unread"],
        "updated_at": thread["updated_at"],
        "title": subject.get("title", ""),
        "subject_type": subject.get("type", ""),
        "subject_url": _subject_html_url(subject.get("url") or ""),
        "repository": thread["repository"]["full_name"],
    }


def _subject_html_url(api_url: str) -> str:
    """Best-effort conversion of a subject API URL to its github.com equivalent."""
    if not api_url:
        return ""
    return api_url.replace(
        "https://api.github.com/repos/", "https://github.com/"
    ).replace("/pulls/", "/pull/")
