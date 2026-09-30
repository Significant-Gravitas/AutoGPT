"""Block that sets up a Capy automation: a standing job that starts agent runs."""

from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._automations_api import CapyAutomationsClient, trigger_payload
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, capy_credentials_field
from ._models import ModelRoute, model_route_field, resolve_model_id
from ._testdata import TEST_AUTOMATION, TEST_PROJECT
from ._types import (
    Automation,
    AutomationTrigger,
    MachineSize,
    ReasoningEffort,
    ThreadMode,
)


class CapyCreateAutomationBlock(Block):
    """Set up a Capy automation that starts agent runs on a schedule or an event."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        project_id: str = SchemaField(
            description=(
                "The Capy project whose repositories the runs work in (see "
                "Capy List Projects)"
            )
        )
        name: str = SchemaField(description="A short name for the automation")
        prompt: str = SchemaField(
            description=(
                "The brief every run starts from, written for its trigger, e.g. "
                "'Root-cause this Sentry issue and open a fix pull request if "
                "it is fixable'"
            )
        )
        trigger: AutomationTrigger = SchemaField(
            description=(
                "What starts a run: schedule (a cron), github, slack, sentry or "
                "linear events, incoming_webhook (a URL you POST to), or "
                "on_demand (started by hand)"
            )
        )
        event: str = SchemaField(
            description=(
                "For github, slack, sentry and linear: the event that starts a "
                "run. github: pull_request_opened, pull_request_merged, checks, "
                "workflow_run, issue_comment, label_change and more. slack: "
                "message or reaction. sentry: any_issue, issue_lifecycle or "
                "event_alert. linear: issue_created or status_changed."
            ),
            default="",
        )
        cron: str = SchemaField(
            description="For schedule: a five-field cron, e.g. 0 9 * * 1-5",
            default="",
        )
        timezone: str = SchemaField(
            description="For schedule: the IANA timezone the cron runs in",
            default="UTC",
            advanced=True,
        )
        run_when: str = SchemaField(
            description=(
                "One plain sentence Capy checks each event against before it "
                "starts a run, e.g. 'Only errors raised by the backend'. Not "
                "used by schedule or on_demand."
            ),
            default="",
        )
        conditions: dict[str, Any] = SchemaField(
            description=(
                "Filters on the event, in Capy's shape for the trigger; an "
                "event passes a filter when it matches any listed value. "
                "github: repositories, branches, labels, authors, conclusions. "
                "sentry: projects, levels. linear: teams, labels, statuses. "
                "slack: channels and users, by id. incoming_webhook: contains, "
                "excludes, regex."
            ),
            default_factory=dict,
            advanced=True,
        )
        max_runs_per_day: int = SchemaField(
            description=(
                "The most runs Capy starts in a day, so a noisy trigger can't "
                "run up the bill"
            ),
            default=10,
            ge=1,
        )
        model_id: str = SchemaField(
            description=(
                "Capy model ID every run uses, e.g. openai/gpt-6-astra, or a "
                "bare name to combine with model_route. Empty lets each run use "
                "its owner's default model."
            ),
            default="",
            advanced=True,
        )
        model_route: ModelRoute = model_route_field()
        reasoning: ReasoningEffort = SchemaField(
            description="Reasoning effort for model_id. Needs model_id.",
            default=ReasoningEffort.DEFAULT,
            advanced=True,
        )
        thread_mode: ThreadMode = SchemaField(
            description=(
                "new starts a thread per run; single keeps every run in one "
                "thread, so each run sees the earlier ones"
            ),
            default=ThreadMode.NEW,
            advanced=True,
        )
        machine_size: MachineSize = SchemaField(
            description="Machine size for each run's VM. Empty uses Capy's default.",
            default=MachineSize.DEFAULT,
            advanced=True,
        )
        description: str = SchemaField(
            description="What the automation is for", default="", advanced=True
        )
        enabled: bool = SchemaField(
            description="Start listening right away. False creates it paused.",
            default=True,
            advanced=True,
        )
        request_id: str = SchemaField(
            description=(
                "Idempotency key. Re-sending the same request_id returns the "
                "automation it already created. Leave empty to generate one."
            ),
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        automation: Automation = SchemaField(description="The created automation")
        automation_id: str = SchemaField(
            description="Pass to Capy Set Automation Enabled or Capy Delete Automation"
        )
        automation_url: str = SchemaField(
            description="The automation in the Capy app, with its runs and settings"
        )
        enabled: bool = SchemaField(
            description="Whether it is listening for its trigger"
        )
        webhook_url: str = SchemaField(
            description=(
                "For an incoming_webhook trigger: POST events here to start "
                "runs. Keep it private, since every request can start a paid run."
            )
        )

    def __init__(self):
        super().__init__(
            id="8e55ced6-141f-4a1b-8223-c878ab156572",
            description=(
                "Sets up a Capy automation: a standing job that starts a Capy "
                "coding agent run on a schedule or when an event arrives from "
                "GitHub, Sentry, Linear, Slack or a webhook, e.g. open a fix "
                "pull request for every new Sentry error. Runs are capped per "
                "day and show up as ordinary Capy threads."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.AGENT},
            input_schema=CapyCreateAutomationBlock.Input,
            output_schema=CapyCreateAutomationBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": TEST_PROJECT.id,
                "name": TEST_AUTOMATION.name,
                "prompt": TEST_AUTOMATION.prompt,
                "trigger": AutomationTrigger.SENTRY,
                "event": "any_issue",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("automation", TEST_AUTOMATION),
                ("automation_id", TEST_AUTOMATION.id),
                ("automation_url", TEST_AUTOMATION.url),
                ("enabled", True),
            ],
            test_mock={"create_automation": lambda *args, **kwargs: TEST_AUTOMATION},
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def create_automation(
        credentials: APIKeyCredentials,
        input_data: "CapyCreateAutomationBlock.Input",
        trigger: dict[str, Any],
    ) -> Automation:
        return await CapyAutomationsClient(credentials).create_automation(
            project_id=input_data.project_id,
            name=input_data.name,
            prompt=input_data.prompt,
            trigger=trigger,
            description=input_data.description,
            model_id=resolve_model_id(input_data.model_id, input_data.model_route),
            reasoning=input_data.reasoning.value,
            thread_mode=input_data.thread_mode.value,
            max_runs_per_day=input_data.max_runs_per_day,
            machine_size=input_data.machine_size.value,
            enabled=input_data.enabled,
            request_id=input_data.request_id,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Built before the call so a missing cron or unknown event fails here
        # with the fix, not as Capy's generic invalid-request error.
        trigger = trigger_payload(
            input_data.trigger,
            event=input_data.event,
            cron=input_data.cron,
            timezone=input_data.timezone,
            conditions=input_data.conditions,
            run_when=input_data.run_when,
        )
        automation = await self.create_automation(credentials, input_data, trigger)
        yield "automation", automation
        yield "automation_id", automation.id
        yield "automation_url", automation.url
        yield "enabled", automation.enabled
        if automation.webhook_url:
            yield "webhook_url", automation.webhook_url
