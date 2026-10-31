"""
Exa Search Monitor run blocks: trigger a run, list runs and get one run.
"""

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

from ._config import exa
from ._exa_rest import QueryParams, add_param, path_segment
from ._search_monitor_samples import SAMPLE_MONITOR_ID, SAMPLE_RUN, SAMPLE_RUN_ID
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .helpers import merge_exa_cost
from .search_monitors_models import (
    MONITORS_URL,
    MonitorGrounding,
    SearchMonitorApiMixin,
    SearchMonitorRunModel,
)

_SAMPLE_RUN_MODEL = SearchMonitorRunModel.from_api(SAMPLE_RUN)


class ExaTriggerSearchMonitorBlock(SearchMonitorApiMixin, Block):
    """Start a Search Monitor run now, outside its schedule."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The monitor to run. Works for active and paused monitors.",
            placeholder=SAMPLE_MONITOR_ID,
        )

    class Output(BlockSchemaOutput):
        monitor_id: str = SchemaField(description="The monitor that was triggered")
        triggered: bool = SchemaField(description="Whether Exa started the run")

    def __init__(self):
        super().__init__(
            id="29b27ff0-0098-4859-90e7-f7190c9204b7",
            description="Run an Exa Search Monitor now; results go to its webhook "
            "and can also be read with List/Get Search Monitor Runs",
            categories={BlockCategory.SEARCH},
            input_schema=ExaTriggerSearchMonitorBlock.Input,
            output_schema=ExaTriggerSearchMonitorBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "monitor_id": SAMPLE_MONITOR_ID,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("monitor_id", SAMPLE_MONITOR_ID), ("triggered", True)],
            test_mock={"call_exa": lambda *args, **kwargs: {"triggered": True}},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials,
            "POST",
            f"{MONITORS_URL}/{path_segment(input_data.monitor_id)}/trigger",
        )
        merge_exa_cost(self, data)
        yield "monitor_id", input_data.monitor_id
        yield "triggered", bool(data.get("triggered", False))


class ExaListSearchMonitorRunsBlock(SearchMonitorApiMixin, Block):
    """List a Search Monitor's runs, newest first."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The monitor whose runs to list", placeholder=SAMPLE_MONITOR_ID
        )
        limit: int = SchemaField(
            default=10, description="Runs per page (1-100)", ge=1, le=100
        )
        cursor: str = SchemaField(
            default="", description="Cursor from a previous page's next_cursor"
        )

    class Output(BlockSchemaOutput):
        runs: list[SearchMonitorRunModel] = SchemaField(
            description="Runs on this page, newest first. Output is filled in "
            "only for completed runs."
        )
        run: SearchMonitorRunModel = SchemaField(
            title="Run", description="Each run, one at a time"
        )
        has_more: bool = SchemaField(description="Whether another page exists")
        next_cursor: str | None = SchemaField(
            description="Pass as cursor to fetch the next page"
        )

    def __init__(self):
        super().__init__(
            id="62d795f3-174f-4ec6-a68c-4747b0e5f99f",
            description="List an Exa Search Monitor's runs, newest first, with "
            "status and output",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListSearchMonitorRunsBlock.Input,
            output_schema=ExaListSearchMonitorRunsBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "monitor_id": SAMPLE_MONITOR_ID,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("runs", [_SAMPLE_RUN_MODEL]),
                ("run", _SAMPLE_RUN_MODEL),
                ("has_more", False),
                ("next_cursor", None),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "data": [SAMPLE_RUN],
                    "hasMore": False,
                    "nextCursor": None,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params: QueryParams = []
        add_param(params, "limit", input_data.limit)
        add_param(params, "cursor", input_data.cursor)
        data = await self.call_exa(
            credentials,
            "GET",
            f"{MONITORS_URL}/{path_segment(input_data.monitor_id)}/runs",
            params=params,
        )
        runs = [SearchMonitorRunModel.from_api(item) for item in data.get("data", [])]
        yield "runs", runs
        for run in runs:
            yield "run", run
        yield "has_more", data.get("hasMore", False)
        yield "next_cursor", data.get("nextCursor")


class ExaGetSearchMonitorRunBlock(SearchMonitorApiMixin, Block):
    """Get one Search Monitor run, including its output once completed."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        monitor_id: str = SchemaField(
            description="The monitor the run belongs to", placeholder=SAMPLE_MONITOR_ID
        )
        run_id: str = SchemaField(
            description="The run to fetch", placeholder=SAMPLE_RUN_ID
        )

    class Output(BlockSchemaOutput):
        run_id: str = SchemaField(description="The run's ID")
        status: str = SchemaField(
            description="pending, running, completed, failed or cancelled"
        )
        results: list[dict[str, Any]] = SchemaField(
            description="New search results the run found (empty until completed)"
        )
        result: dict[str, Any] = SchemaField(description="Each result, one at a time")
        content: Any = SchemaField(
            description="Synthesized findings: text, or JSON matching the "
            "monitor's output schema"
        )
        grounding: list[MonitorGrounding] = SchemaField(
            description="Citations and confidence for each synthesized field"
        )
        fail_reason: str | None = SchemaField(
            description="Why the run failed, e.g. insufficient_credits or "
            "source_not_available; empty unless failed"
        )
        run: SearchMonitorRunModel = SchemaField(
            title="Run", description="The full run"
        )

    def __init__(self):
        super().__init__(
            id="a5164551-58d2-4cd3-96c6-82045720cd7d",
            description="Get an Exa Search Monitor run's status, new results and "
            "synthesized findings",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetSearchMonitorRunBlock.Input,
            output_schema=ExaGetSearchMonitorRunBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "monitor_id": SAMPLE_MONITOR_ID,
                "run_id": SAMPLE_RUN_ID,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("run_id", SAMPLE_RUN_ID),
                ("status", "completed"),
                ("results", _SAMPLE_RUN_MODEL.results),
                ("result", _SAMPLE_RUN_MODEL.results[0]),
                ("content", _SAMPLE_RUN_MODEL.content),
                ("grounding", _SAMPLE_RUN_MODEL.grounding),
                ("fail_reason", None),
                ("run", _SAMPLE_RUN_MODEL),
            ],
            test_mock={"call_exa": lambda *args, **kwargs: SAMPLE_RUN},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials,
            "GET",
            f"{MONITORS_URL}/{path_segment(input_data.monitor_id)}/runs/{path_segment(input_data.run_id)}",
        )
        merge_exa_cost(self, data)
        run = SearchMonitorRunModel.from_api(data)
        yield "run_id", run.id
        yield "status", run.status
        yield "results", run.results
        for result in run.results:
            yield "result", result
        yield "content", run.content
        yield "grounding", run.grounding
        yield "fail_reason", run.fail_reason
        yield "run", run
