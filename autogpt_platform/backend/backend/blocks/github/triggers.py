import json
import logging
from pathlib import Path

from pydantic import BaseModel

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockWebhookConfig,
)
from backend.data.model import SchemaField
from backend.integrations.providers import ProviderName

from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentialsField,
    GithubCredentialsInput,
)

logger = logging.getLogger(__name__)


# --8<-- [start:GithubTriggerExample]
class GitHubTriggerBase:
    class Input(BlockSchemaInput):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description=(
                "Repository to subscribe to.\n\n"
                "**Note:** Make sure your GitHub credentials have permissions "
                "to create webhooks on this repo."
            ),
            placeholder="{owner}/{repo}",
        )
        # --8<-- [start:example-payload-field]
        payload: dict = SchemaField(hidden=True, default_factory=dict)
        # --8<-- [end:example-payload-field]

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(
            description="The complete webhook payload that was received from GitHub. "
            "Includes information about the affected resource (e.g. pull request), "
            "the event, and the user who triggered the event."
        )
        triggered_by_user: dict = SchemaField(
            description="Object representing the GitHub user who triggered the event"
        )
        error: str = SchemaField(
            description="Error message if the payload could not be processed"
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "payload", input_data.payload
        yield "triggered_by_user", input_data.payload["sender"]


class GithubPullRequestTriggerBlock(GitHubTriggerBase, Block):
    EXAMPLE_PAYLOAD_FILE = (
        Path(__file__).parent / "example_payloads" / "pull_request.synchronize.json"
    )

    # --8<-- [start:example-event-filter]
    class Input(GitHubTriggerBase.Input):
        class EventsFilter(BaseModel):
            """
            https://docs.github.com/en/webhooks/webhook-events-and-payloads#pull_request
            """

            opened: bool = False
            edited: bool = False
            closed: bool = False
            reopened: bool = False
            synchronize: bool = False
            assigned: bool = False
            unassigned: bool = False
            labeled: bool = False
            unlabeled: bool = False
            converted_to_draft: bool = False
            locked: bool = False
            unlocked: bool = False
            enqueued: bool = False
            dequeued: bool = False
            milestoned: bool = False
            demilestoned: bool = False
            ready_for_review: bool = False
            review_requested: bool = False
            review_request_removed: bool = False
            auto_merge_enabled: bool = False
            auto_merge_disabled: bool = False

        events: EventsFilter = SchemaField(
            title="Events", description="The events to subscribe to"
        )
        # --8<-- [end:example-event-filter]

    class Output(GitHubTriggerBase.Output):
        event: str = SchemaField(
            description="The PR event that triggered the webhook (e.g. 'opened')"
        )
        number: int = SchemaField(description="The number of the affected pull request")
        pull_request: dict = SchemaField(
            description="Object representing the affected pull request"
        )
        pull_request_url: str = SchemaField(
            description="The URL of the affected pull request"
        )

    def __init__(self):
        from backend.integrations.webhooks.github import GithubWebhookType

        example_payload = json.loads(
            self.EXAMPLE_PAYLOAD_FILE.read_text(encoding="utf-8")
        )

        super().__init__(
            id="6c60ec01-8128-419e-988f-96a063ee2fea",
            description="This block triggers on pull request events and outputs the event type and payload.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=GithubPullRequestTriggerBlock.Input,
            output_schema=GithubPullRequestTriggerBlock.Output,
            # --8<-- [start:example-webhook_config]
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.GITHUB,
                webhook_type=GithubWebhookType.REPO,
                resource_format="{repo}",
                event_filter_input="events",
                event_format="pull_request.{event}",
            ),
            # --8<-- [end:example-webhook_config]
            test_input={
                "repo": "Significant-Gravitas/AutoGPT",
                "events": {"opened": True, "synchronize": True},
                "credentials": TEST_CREDENTIALS_INPUT,
                "payload": example_payload,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", example_payload),
                ("triggered_by_user", example_payload["sender"]),
                ("event", example_payload["action"]),
                ("number", example_payload["number"]),
                ("pull_request", example_payload["pull_request"]),
                ("pull_request_url", example_payload["pull_request"]["html_url"]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:  # type: ignore
        async for name, value in super().run(input_data, **kwargs):
            yield name, value
        yield "event", input_data.payload["action"]
        yield "number", input_data.payload["number"]
        yield "pull_request", input_data.payload["pull_request"]
        yield "pull_request_url", input_data.payload["pull_request"]["html_url"]


# --8<-- [end:GithubTriggerExample]


class GithubStarTriggerBlock(GitHubTriggerBase, Block):
    """Trigger block for GitHub star events - useful for milestone celebrations."""

    EXAMPLE_PAYLOAD_FILE = (
        Path(__file__).parent / "example_payloads" / "star.created.json"
    )

    class Input(GitHubTriggerBase.Input):
        class EventsFilter(BaseModel):
            """
            https://docs.github.com/en/webhooks/webhook-events-and-payloads#star
            """

            created: bool = False
            deleted: bool = False

        events: EventsFilter = SchemaField(
            title="Events", description="The star events to subscribe to"
        )

    class Output(GitHubTriggerBase.Output):
        event: str = SchemaField(
            description="The star event that triggered the webhook ('created' or 'deleted')"
        )
        starred_at: str = SchemaField(
            description="ISO timestamp when the repo was starred (empty if deleted)"
        )
        stargazers_count: int = SchemaField(
            description="Current number of stars on the repository"
        )
        repository_name: str = SchemaField(
            description="Full name of the repository (owner/repo)"
        )
        repository_url: str = SchemaField(description="URL to the repository")

    def __init__(self):
        from backend.integrations.webhooks.github import GithubWebhookType

        example_payload = json.loads(
            self.EXAMPLE_PAYLOAD_FILE.read_text(encoding="utf-8")
        )

        super().__init__(
            id="551e0a35-100b-49b7-89b8-3031322239b6",
            description="This block triggers on GitHub star events. "
            "Useful for celebrating milestones (e.g., 1k, 10k stars) or tracking engagement.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=GithubStarTriggerBlock.Input,
            output_schema=GithubStarTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.GITHUB,
                webhook_type=GithubWebhookType.REPO,
                resource_format="{repo}",
                event_filter_input="events",
                event_format="star.{event}",
            ),
            test_input={
                "repo": "Significant-Gravitas/AutoGPT",
                "events": {"created": True},
                "credentials": TEST_CREDENTIALS_INPUT,
                "payload": example_payload,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", example_payload),
                ("triggered_by_user", example_payload["sender"]),
                ("event", example_payload["action"]),
                ("starred_at", example_payload.get("starred_at", "")),
                ("stargazers_count", example_payload["repository"]["stargazers_count"]),
                ("repository_name", example_payload["repository"]["full_name"]),
                ("repository_url", example_payload["repository"]["html_url"]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:  # type: ignore
        async for name, value in super().run(input_data, **kwargs):
            yield name, value
        yield "event", input_data.payload["action"]
        yield "starred_at", input_data.payload.get("starred_at", "")
        yield "stargazers_count", input_data.payload["repository"]["stargazers_count"]
        yield "repository_name", input_data.payload["repository"]["full_name"]
        yield "repository_url", input_data.payload["repository"]["html_url"]


class GithubReleaseTriggerBlock(GitHubTriggerBase, Block):
    """Trigger block for GitHub release events - ideal for announcing new versions."""

    EXAMPLE_PAYLOAD_FILE = (
        Path(__file__).parent / "example_payloads" / "release.published.json"
    )

    class Input(GitHubTriggerBase.Input):
        class EventsFilter(BaseModel):
            """
            https://docs.github.com/en/webhooks/webhook-events-and-payloads#release
            """

            published: bool = False
            unpublished: bool = False
            created: bool = False
            edited: bool = False
            deleted: bool = False
            prereleased: bool = False
            released: bool = False

        events: EventsFilter = SchemaField(
            title="Events", description="The release events to subscribe to"
        )

    class Output(GitHubTriggerBase.Output):
        event: str = SchemaField(
            description="The release event that triggered the webhook (e.g., 'published')"
        )
        release: dict = SchemaField(description="The full release object")
        release_url: str = SchemaField(description="URL to the release page")
        tag_name: str = SchemaField(description="The release tag name (e.g., 'v1.0.0')")
        release_name: str = SchemaField(description="Human-readable release name")
        body: str = SchemaField(description="Release notes/description")
        prerelease: bool = SchemaField(description="Whether this is a prerelease")
        draft: bool = SchemaField(description="Whether this is a draft release")
        assets: list = SchemaField(description="List of release assets/files")

    def __init__(self):
        from backend.integrations.webhooks.github import GithubWebhookType

        example_payload = json.loads(
            self.EXAMPLE_PAYLOAD_FILE.read_text(encoding="utf-8")
        )

        super().__init__(
            id="2052dd1b-74e1-46ac-9c87-c7a0e057b60b",
            description="This block triggers on GitHub release events. "
            "Perfect for automating announcements to Discord, Twitter, or other platforms.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=GithubReleaseTriggerBlock.Input,
            output_schema=GithubReleaseTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.GITHUB,
                webhook_type=GithubWebhookType.REPO,
                resource_format="{repo}",
                event_filter_input="events",
                event_format="release.{event}",
            ),
            test_input={
                "repo": "Significant-Gravitas/AutoGPT",
                "events": {"published": True},
                "credentials": TEST_CREDENTIALS_INPUT,
                "payload": example_payload,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", example_payload),
                ("triggered_by_user", example_payload["sender"]),
                ("event", example_payload["action"]),
                ("release", example_payload["release"]),
                ("release_url", example_payload["release"]["html_url"]),
                ("tag_name", example_payload["release"]["tag_name"]),
                ("release_name", example_payload["release"]["name"]),
                ("body", example_payload["release"]["body"]),
                ("prerelease", example_payload["release"]["prerelease"]),
                ("draft", example_payload["release"]["draft"]),
                ("assets", example_payload["release"]["assets"]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:  # type: ignore
        async for name, value in super().run(input_data, **kwargs):
            yield name, value
        release = input_data.payload["release"]
        yield "event", input_data.payload["action"]
        yield "release", release
        yield "release_url", release["html_url"]
        yield "tag_name", release["tag_name"]
        yield "release_name", release.get("name", "")
        yield "body", release.get("body", "")
        yield "prerelease", release["prerelease"]
        yield "draft", release["draft"]
        yield "assets", release["assets"]


class GithubIssuesTriggerBlock(GitHubTriggerBase, Block):
    """Trigger block for GitHub issues events - great for triage and notifications."""

    EXAMPLE_PAYLOAD_FILE = (
        Path(__file__).parent / "example_payloads" / "issues.opened.json"
    )

    class Input(GitHubTriggerBase.Input):
        class EventsFilter(BaseModel):
            """
            https://docs.github.com/en/webhooks/webhook-events-and-payloads#issues
            """

            opened: bool = False
            edited: bool = False
            deleted: bool = False
            closed: bool = False
            reopened: bool = False
            assigned: bool = False
            unassigned: bool = False
            labeled: bool = False
            unlabeled: bool = False
            locked: bool = False
            unlocked: bool = False
            transferred: bool = False
            milestoned: bool = False
            demilestoned: bool = False
            pinned: bool = False
            unpinned: bool = False

        events: EventsFilter = SchemaField(
            title="Events", description="The issue events to subscribe to"
        )

    class Output(GitHubTriggerBase.Output):
        event: str = SchemaField(
            description="The issue event that triggered the webhook (e.g., 'opened')"
        )
        number: int = SchemaField(description="The issue number")
        issue: dict = SchemaField(description="The full issue object")
        issue_url: str = SchemaField(description="URL to the issue")
        issue_title: str = SchemaField(description="The issue title")
        issue_body: str = SchemaField(description="The issue body/description")
        labels: list = SchemaField(description="List of labels on the issue")
        assignees: list = SchemaField(description="List of assignees")
        state: str = SchemaField(description="Issue state ('open' or 'closed')")

    def __init__(self):
        from backend.integrations.webhooks.github import GithubWebhookType

        example_payload = json.loads(
            self.EXAMPLE_PAYLOAD_FILE.read_text(encoding="utf-8")
        )

        super().__init__(
            id="b2605464-e486-4bf4-aad3-d8a213c8a48a",
            description="This block triggers on GitHub issues events. "
            "Useful for automated triage, notifications, and welcoming first-time contributors.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=GithubIssuesTriggerBlock.Input,
            output_schema=GithubIssuesTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.GITHUB,
                webhook_type=GithubWebhookType.REPO,
                resource_format="{repo}",
                event_filter_input="events",
                event_format="issues.{event}",
            ),
            test_input={
                "repo": "Significant-Gravitas/AutoGPT",
                "events": {"opened": True},
                "credentials": TEST_CREDENTIALS_INPUT,
                "payload": example_payload,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", example_payload),
                ("triggered_by_user", example_payload["sender"]),
                ("event", example_payload["action"]),
                ("number", example_payload["issue"]["number"]),
                ("issue", example_payload["issue"]),
                ("issue_url", example_payload["issue"]["html_url"]),
                ("issue_title", example_payload["issue"]["title"]),
                ("issue_body", example_payload["issue"]["body"]),
                ("labels", example_payload["issue"]["labels"]),
                ("assignees", example_payload["issue"]["assignees"]),
                ("state", example_payload["issue"]["state"]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:  # type: ignore
        async for name, value in super().run(input_data, **kwargs):
            yield name, value
        issue = input_data.payload["issue"]
        yield "event", input_data.payload["action"]
        yield "number", issue["number"]
        yield "issue", issue
        yield "issue_url", issue["html_url"]
        yield "issue_title", issue["title"]
        yield "issue_body", issue.get("body") or ""
        yield "labels", issue["labels"]
        yield "assignees", issue["assignees"]
        yield "state", issue["state"]


class GithubDiscussionTriggerBlock(GitHubTriggerBase, Block):
    """Trigger block for GitHub discussion events - perfect for community Q&A sync."""

    EXAMPLE_PAYLOAD_FILE = (
        Path(__file__).parent / "example_payloads" / "discussion.created.json"
    )

    class Input(GitHubTriggerBase.Input):
        class EventsFilter(BaseModel):
            """
            https://docs.github.com/en/webhooks/webhook-events-and-payloads#discussion
            """

            created: bool = False
            edited: bool = False
            deleted: bool = False
            answered: bool = False
            unanswered: bool = False
            labeled: bool = False
            unlabeled: bool = False
            locked: bool = False
            unlocked: bool = False
            category_changed: bool = False
            transferred: bool = False
            pinned: bool = False
            unpinned: bool = False

        events: EventsFilter = SchemaField(
            title="Events", description="The discussion events to subscribe to"
        )

    class Output(GitHubTriggerBase.Output):
        event: str = SchemaField(
            description="The discussion event that triggered the webhook"
        )
        number: int = SchemaField(description="The discussion number")
        discussion: dict = SchemaField(description="The full discussion object")
        discussion_url: str = SchemaField(description="URL to the discussion")
        title: str = SchemaField(description="The discussion title")
        body: str = SchemaField(description="The discussion body")
        category: dict = SchemaField(description="The discussion category object")
        category_name: str = SchemaField(description="Name of the category")
        state: str = SchemaField(description="Discussion state")

    def __init__(self):
        from backend.integrations.webhooks.github import GithubWebhookType

        example_payload = json.loads(
            self.EXAMPLE_PAYLOAD_FILE.read_text(encoding="utf-8")
        )

        super().__init__(
            id="87f847b3-d81a-424e-8e89-acadb5c9d52b",
            description="This block triggers on GitHub Discussions events. "
            "Great for syncing Q&A to Discord or auto-responding to common questions. "
            "Note: Discussions must be enabled on the repository.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=GithubDiscussionTriggerBlock.Input,
            output_schema=GithubDiscussionTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.GITHUB,
                webhook_type=GithubWebhookType.REPO,
                resource_format="{repo}",
                event_filter_input="events",
                event_format="discussion.{event}",
            ),
            test_input={
                "repo": "Significant-Gravitas/AutoGPT",
                "events": {"created": True},
                "credentials": TEST_CREDENTIALS_INPUT,
                "payload": example_payload,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", example_payload),
                ("triggered_by_user", example_payload["sender"]),
                ("event", example_payload["action"]),
                ("number", example_payload["discussion"]["number"]),
                ("discussion", example_payload["discussion"]),
                ("discussion_url", example_payload["discussion"]["html_url"]),
                ("title", example_payload["discussion"]["title"]),
                ("body", example_payload["discussion"]["body"]),
                ("category", example_payload["discussion"]["category"]),
                ("category_name", example_payload["discussion"]["category"]["name"]),
                ("state", example_payload["discussion"]["state"]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:  # type: ignore
        async for name, value in super().run(input_data, **kwargs):
            yield name, value
        discussion = input_data.payload["discussion"]
        yield "event", input_data.payload["action"]
        yield "number", discussion["number"]
        yield "discussion", discussion
        yield "discussion_url", discussion["html_url"]
        yield "title", discussion["title"]
        yield "body", discussion.get("body") or ""
        yield "category", discussion["category"]
        yield "category_name", discussion["category"]["name"]
        yield "state", discussion["state"]
