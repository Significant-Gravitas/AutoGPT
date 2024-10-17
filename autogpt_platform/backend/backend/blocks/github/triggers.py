import logging
from typing import Optional

import requests
from pydantic import BaseModel

from backend.data.block import (
    Block,
    BlockCategory,
    BlockInput,
    BlockOutput,
    BlockSchema,
)
from backend.data.model import SchemaField

from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GithubCredentials,
    GithubCredentialsField,
    GithubCredentialsInput,
)

logger = logging.getLogger(__name__)


class GitHubBaseTriggerBlock(Block):
    class Input(BlockSchema):
        credentials: GithubCredentialsInput = GithubCredentialsField("repo")
        repo: str = SchemaField(
            description="Repository to subscribe to",
            placeholder="{owner}/{repo}",
        )
        payload: dict = SchemaField(description="Webhook payload", exclude=True)

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

    @classmethod
    def create_webhook(
        cls, credentials: GithubCredentials, repo: str, events: list[str]
    ):
        # TODO: Create webhook in DB

        # Create webhook on GitHub
        api_url = f"https://api.github.com/repos/{repo}/hooks"
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }
        payload = {
            "name": "web",
            "active": True,
            "events": events,
            "config": {
                "url": "YOUR_WEBHOOK_URL",  # Replace with actual webhook URL
                "content_type": "json",
                "insecure_ssl": "0",
            },
        }
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()

    @classmethod
    def update_webhook(
        cls,
        credentials: GithubCredentials,
        repo: str,
        events: list[str],
        webhook_id: str,
    ):
        # TODO: Update webhook in DB

        # Update webhook on GitHub
        api_url = f"https://api.github.com/repos/{repo}/hooks/{webhook_id}"
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }
        payload = {
            "active": True,
            "events": events,
            "config": {
                "url": "YOUR_WEBHOOK_URL",  # Replace with actual webhook URL
                "content_type": "json",
                "insecure_ssl": "0",
            },
        }
        response = requests.patch(api_url, headers=headers, json=payload)
        response.raise_for_status()

    @classmethod
    def delete_webhook(cls, webhook_id: str):
        # TODO: Delete webhook from DB
        pass

    @classmethod
    def deregister_webhook(
        cls, credentials: GithubCredentials, repo: str, github_webhook_id: str
    ):
        api_url = f"https://api.github.com/repos/{repo}/hooks/{github_webhook_id}"
        headers = {
            "Authorization": credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }
        response = requests.delete(api_url, headers=headers)
        response.raise_for_status()


class GithubPullRequestTriggerBlock(GitHubBaseTriggerBlock):
    class Input(GitHubBaseTriggerBlock.Input):
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

    class Output(GitHubBaseTriggerBlock.Output):
        number: int = SchemaField(description="The number of the affected pull request")
        pull_request: dict = SchemaField(
            description="Object representing the pull request"
        )

    def __init__(self):
        super().__init__(
            id="6c60ec01-8128-419e-988f-96a063ee2fea",
            description="This block triggers on pull request events and outputs the event type and payload.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=GithubPullRequestTriggerBlock.Input,
            output_schema=GithubPullRequestTriggerBlock.Output,
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
        input_data: Input,  # type: ignore
        **kwargs,
    ) -> BlockOutput:
        super().run(input_data, **kwargs)
        yield "number", input_data.payload["number"]
        yield "pull_request", input_data.payload["pull_request"]

    def on_node_update(
        self,
        new_preset_inputs: BlockInput,
        old_preset_inputs: Optional[BlockInput] = None,
        *,
        new_credentials: Optional[GithubCredentials] = None,
        old_credentials: Optional[GithubCredentials] = None,
    ) -> None:
        old_has_all = old_preset_inputs and all(
            key in old_preset_inputs for key in ["credentials", "repo", "events"]
        )
        new_has_all = all(
            key in new_preset_inputs for key in ["credentials", "repo", "events"]
        )

        if new_has_all and new_credentials and old_preset_inputs and old_has_all:
            # Input was and is complete -> update webhook to new config

            # TODO: Get webhook_id from DB
            webhook_id = "WEBHOOK_ID"  # FIXME: Replace with actual webhook ID
            github_webhook_id = "GITHUB_WEBHOOK_ID"  # FIXME

            if new_credentials != old_credentials:
                # Credentials were replaced -> recreate webhook with new credentials
                if old_credentials:
                    self.deregister_webhook(
                        old_credentials,
                        old_preset_inputs["repo"],
                        github_webhook_id,
                    )
                else:
                    logger.warning(
                        f"Cannot deregister webhook #{webhook_id} with unavailable "
                        f"credentials #{old_preset_inputs['credentials']['id']} "
                        f"(GitHub webhook ID: {github_webhook_id})"
                    )
                self.delete_webhook(webhook_id)
                self.create_webhook(
                    new_credentials,
                    new_preset_inputs["repo"],
                    new_preset_inputs["events"],
                )
            else:
                self.update_webhook(
                    new_credentials,
                    new_preset_inputs["repo"],
                    new_preset_inputs["events"],
                    webhook_id,
                )
        elif new_has_all and new_credentials and not old_has_all:
            # Input was incomplete -> create new webhook
            self.create_webhook(
                new_credentials,
                new_preset_inputs["repo"],
                new_preset_inputs["events"],
            )
        elif not new_has_all and old_preset_inputs and old_has_all:
            # Input has become incomplete -> delete webhook
            self.on_node_delete(old_preset_inputs, credentials=old_credentials)

    def on_node_delete(
        self,
        preset_inputs: BlockInput,
        *,
        credentials: Optional[GithubCredentials] = None,
    ) -> None:
        # TODO: Get webhook_id from DB
        webhook_id = "WEBHOOK_ID"  # FIXME: Replace with actual webhook ID
        github_webhook_id = "GITHUB_WEBHOOK_ID"  # FIXME

        if all(key in preset_inputs for key in ["credentials", "repo", "events"]):
            if credentials:
                self.deregister_webhook(
                    credentials,
                    preset_inputs["repo"],
                    github_webhook_id,
                )
            else:
                logger.warning(
                    f"Cannot deregister webhook #{webhook_id} with "
                    f"unavailable credentials #{preset_inputs['credentials']['id']} "
                    f"(GitHub webhook ID: {github_webhook_id})"
                )

            self.delete_webhook(webhook_id)
