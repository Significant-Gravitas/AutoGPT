import json
import logging
from pathlib import Path

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


# --8<-- [start:GithubTriggerExample]
class GitHubTriggerBase:
    class Input(BlockSchema):
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
        payload: dict = SchemaField(hidden=True, default={})
        # --8<-- [end:example-payload-field]

    class Output(BlockSchema):
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

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
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
                provider="github",
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

    def run(self, input_data: Input, **kwargs) -> BlockOutput:  # type: ignore
        yield from super().run(input_data, **kwargs)
        yield "event", input_data.payload["action"]
        yield "number", input_data.payload["number"]
        yield "pull_request", input_data.payload["pull_request"]
        yield "pull_request_url", input_data.payload["pull_request"]["html_url"]


# --8<-- [end:GithubTriggerExample]
