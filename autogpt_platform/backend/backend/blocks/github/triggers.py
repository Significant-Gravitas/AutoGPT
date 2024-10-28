import logging

from pydantic import BaseModel

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockWebhookConfig,
)
from backend.data.model import SchemaField

from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentialsField,
    GithubCredentialsInput,
)

logger = logging.getLogger(__name__)


class GitHubTriggerBase:
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description="Repository to subscribe to",
            placeholder="{owner}/{repo}",
        )
        payload: dict = SchemaField(hidden=True, default={})

    class Output(BlockSchema):
        event: str = SchemaField(description="The event that triggered the webhook")
        payload: dict = SchemaField(description="Full payload of the event")
        sender: dict = SchemaField(
            description="Object representing the user who triggered the event"
        )
        error: str = SchemaField(
            description="Error message if the payload could not be processed"
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "event", input_data.payload["action"]
        yield "payload", input_data.payload
        yield "sender", input_data.payload["sender"]


class GithubPullRequestTriggerBlock(GitHubTriggerBase, Block):
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

        events: EventsFilter = SchemaField(description="The events to subscribe to")

    class Output(GitHubTriggerBase.Output):
        number: int = SchemaField(description="The number of the affected pull request")
        pull_request: dict = SchemaField(
            description="Object representing the pull request"
        )

    def __init__(self):
        from backend.integrations.providers import ProviderName
        from backend.integrations.webhooks.github import GithubWebhookType

        super().__init__(
            id="6c60ec01-8128-419e-988f-96a063ee2fea",
            description="This block triggers on pull request events and outputs the event type and payload.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=GithubPullRequestTriggerBlock.Input,
            output_schema=GithubPullRequestTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.GITHUB,
                webhook_type=GithubWebhookType.REPO,
                resource_format="{repo}",
                event_filter_input="events",
                event_format="pull_request.{event}",
            ),
            test_input={
                "repo": "owner/repo",
                "events": {"opened": True, "synchronize": True},
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                # ("title", "Title of the pull request"),
                # ("body", "This is the body of the pull request."),
                # ("author", "username"),
                # ("changes", "List of changes made in the pull request."),
            ],
            test_mock={
                # "read_pr": lambda *args, **kwargs: (
                #     "Title of the pull request",
                #     "This is the body of the pull request.",
                #     "username",
                # ),
                # "read_pr_changes": lambda *args, **kwargs: "List of changes made in the pull request.",
            },
        )

    def run(
        self,
        input_data: GitHubTriggerBase.Input,
        **kwargs,
    ) -> BlockOutput:
        super().run(input_data, **kwargs)
        yield "number", input_data.payload["number"]
        yield "pull_request", input_data.payload["pull_request"]
