"""
Exa Websets Monitor Run Blocks

A Websets monitor (``/v0/monitors``) runs a search on its cron schedule; each
execution is a *run*. These blocks list a monitor's runs and fetch one run, so
a workflow can check whether the last scheduled update succeeded.
"""

from typing import Any

from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .websets_api import ExaWebsetsBlockBase, segment
from .websets_fixtures import MONITOR_ID, RUN_ID, sample_monitor_run


class WebsetMonitorRunModel(BaseModel):
    """One execution of a Websets monitor."""

    id: str
    monitor_id: str
    status: str = SchemaField(
        description="created, running, completed, canceled or failed"
    )
    type: str = SchemaField(description="What the run did: search or refresh")
    failed_reason: str | None = None
    created_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None
    failed_at: str | None = None
    canceled_at: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "WebsetMonitorRunModel":
        return cls(
            id=data.get("id", ""),
            monitor_id=data.get("monitorId", ""),
            status=data.get("status", ""),
            type=data.get("type", ""),
            failed_reason=data.get("failedReason"),
            created_at=data.get("createdAt") or "",
            updated_at=data.get("updatedAt") or "",
            completed_at=data.get("completedAt"),
            failed_at=data.get("failedAt"),
            canceled_at=data.get("canceledAt"),
        )


class ExaListWebsetMonitorRunsBlock(ExaWebsetsBlockBase):
    """List the runs of a Websets monitor."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The ID of the Websets monitor whose runs to list",
            placeholder="monitor-id",
        )

    class Output(BlockSchemaOutput):
        runs: list[WebsetMonitorRunModel] = SchemaField(
            description="The monitor's runs"
        )
        run: WebsetMonitorRunModel = SchemaField(
            title="Run", description="Individual run (yielded for each run in the list)"
        )
        has_more: bool = SchemaField(
            description="Whether Exa has more runs than it returned"
        )
        next_cursor: str | None = SchemaField(
            description="Cursor Exa returned for the next page, if any"
        )

    def __init__(self):
        super().__init__(
            id="e9994fbd-7fa2-4bc6-a837-a3ac4fd9efb9",
            description="List the scheduled runs of a Websets monitor, to check whether its updates succeeded",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListWebsetMonitorRunsBlock.Input,
            output_schema=ExaListWebsetMonitorRunsBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "monitor_id": MONITOR_ID,
            },
            test_output=[
                ("runs", [WebsetMonitorRunModel.from_api(sample_monitor_run())]),
                ("run", WebsetMonitorRunModel.from_api(sample_monitor_run())),
                ("has_more", False),
                ("next_cursor", None),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_api": lambda *args, **kwargs: {
                    "data": [sample_monitor_run()],
                    "hasMore": False,
                    "nextCursor": None,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        response = await self._api(
            credentials, "GET", f"/monitors/{segment(input_data.monitor_id)}/runs"
        )
        runs = [WebsetMonitorRunModel.from_api(r) for r in response.get("data", [])]

        yield "runs", runs
        for run in runs:
            yield "run", run
        yield "has_more", bool(response.get("hasMore"))
        yield "next_cursor", response.get("nextCursor")


class ExaGetWebsetMonitorRunBlock(ExaWebsetsBlockBase):
    """Get one run of a Websets monitor."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The ID of the Websets monitor the run belongs to",
            placeholder="monitor-id",
        )
        run_id: str = SchemaField(
            description="The ID of the run to retrieve",
            placeholder="run-id",
        )

    class Output(BlockSchemaOutput):
        run: WebsetMonitorRunModel = SchemaField(
            title="Run", description="The full run details"
        )
        run_id: str = SchemaField(description="The unique identifier for the run")
        status: str = SchemaField(
            description="created, running, completed, canceled or failed"
        )
        type: str = SchemaField(description="What the run did: search or refresh")
        failed_reason: str | None = SchemaField(
            description="Why the run failed (only for failed runs)"
        )
        completed_at: str | None = SchemaField(
            description="When the run completed, if it has"
        )
        created_at: str = SchemaField(description="When the run started")

    def __init__(self):
        super().__init__(
            id="979f5dfc-37c8-4372-b401-a62eb8fb20f3",
            description="Get the status and outcome of one scheduled run of a Websets monitor",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetMonitorRunBlock.Input,
            output_schema=ExaGetWebsetMonitorRunBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "monitor_id": MONITOR_ID,
                "run_id": RUN_ID,
            },
            test_output=[
                ("run", WebsetMonitorRunModel.from_api(sample_monitor_run())),
                ("run_id", RUN_ID),
                ("status", "completed"),
                ("type", "search"),
                ("failed_reason", None),
                ("completed_at", "2024-01-01T00:00:00Z"),
                ("created_at", "2024-01-01T00:00:00Z"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *args, **kwargs: sample_monitor_run()},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        path = (
            f"/monitors/{segment(input_data.monitor_id)}"
            f"/runs/{segment(input_data.run_id)}"
        )
        run = WebsetMonitorRunModel.from_api(await self._api(credentials, "GET", path))

        yield "run", run
        yield "run_id", run.id
        yield "status", run.status
        yield "type", run.type
        yield "failed_reason", run.failed_reason
        yield "completed_at", run.completed_at
        yield "created_at", run.created_at
