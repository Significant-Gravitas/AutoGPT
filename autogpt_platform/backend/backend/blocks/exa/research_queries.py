"""
Legacy Research blocks that read a task: Get Research and Wait for Research.

Both now read Exa Agent runs; see research.py for the migration notes.
"""

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

from . import agent_api
from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .agent_api import AgentWaitResult
from .agent_billing import bill_finished_run
from .agent_fixtures import SAMPLE_COMPLETED_RUN, SAMPLE_EVENT_LIST, SAMPLE_RUN_ID
from .agent_models import AgentRun, AgentRunEvent, parse_event, parse_run
from .research_models import ResearchTaskModel, legacy_finished_outputs, legacy_status

_COMPLETED = ResearchTaskModel.from_agent_run(parse_run(SAMPLE_COMPLETED_RUN))


class ExaGetResearchBlock(Block):
    """Get the status and results of a research task (an Exa Agent run)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        research_id: str = SchemaField(
            description="The ID of the research task to retrieve (an Exa Agent run ID)",
            placeholder="agent_run_01jszdfs0052sg4jc552sg4jc5",
        )
        include_events: bool = SchemaField(
            default=False,
            description="Include detailed event log of research operations",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        research_id: str = SchemaField(description="The research task identifier")
        status: str = SchemaField(
            description="Current status: pending, running, completed, canceled, or failed"
        )
        instructions: str = SchemaField(
            description="The original research instructions"
        )
        model: str = SchemaField(
            description="The research model used (or the Exa Agent effort, for runs "
            "started outside the Research blocks)"
        )
        created_at: int = SchemaField(
            description="When research was created (Unix timestamp in ms)"
        )
        finished_at: int | None = SchemaField(
            description="When research finished (Unix timestamp in ms, if completed/canceled/failed)"
        )
        output_content: str | None = SchemaField(
            description="Research output as text (if completed)"
        )
        output_parsed: dict | None = SchemaField(
            description="Structured JSON output matching outputSchema (if provided and completed)"
        )
        cost_total: float | None = SchemaField(
            description="Total cost in USD (if completed)"
        )
        cost_searches: int | None = SchemaField(
            description="Number of searches performed (if completed)"
        )
        cost_pages: int | None = SchemaField(
            description="Legacy: always 0, because Exa Agent does not report "
            "pages crawled"
        )
        cost_reasoning_tokens: int | None = SchemaField(
            description="Legacy: always 0, because Exa Agent does not report "
            "reasoning tokens"
        )
        error_message: str | None = SchemaField(
            description="Error message if research failed"
        )
        events: list[dict] | None = SchemaField(
            description="Detailed event log (if include_events was True)"
        )

    def __init__(self):
        super().__init__(
            id="b2e3f4a5-6789-4bcd-9012-3456789abcde",
            description="Get status and results of a research task. Legacy: now "
            "reads Exa Agent runs; use 'Get Exa Agent Run' for new workflows.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetResearchBlock.Input,
            output_schema=ExaGetResearchBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "research_id": SAMPLE_RUN_ID,
                "include_events": True,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("research_id", SAMPLE_RUN_ID),
                ("status", "completed"),
                ("instructions", _COMPLETED.instructions),
                ("model", "exa-research"),
                ("created_at", _COMPLETED.created_at),
                ("finished_at", _COMPLETED.finished_at),
                ("output_content", SAMPLE_COMPLETED_RUN["output"]["text"]),
                ("output_parsed", SAMPLE_COMPLETED_RUN["output"]["structured"]),
                ("cost_total", 0.1),
                ("cost_searches", 4),
                ("cost_pages", 0),
                ("cost_reasoning_tokens", 0),
                ("error_message", None),
                ("events", list),
            ],
            test_mock={
                "get_run": lambda *args, **kwargs: SAMPLE_COMPLETED_RUN,
                "list_events": lambda *args, **kwargs: [
                    parse_event(e) for e in SAMPLE_EVENT_LIST["data"]
                ],
            },
        )

    @staticmethod
    async def get_run(credentials: APIKeyCredentials, run_id: str) -> dict:
        return await agent_api.get_agent_run(credentials, run_id)

    @staticmethod
    async def list_events(
        credentials: APIKeyCredentials, run_id: str
    ) -> list[AgentRunEvent]:
        return await agent_api.list_all_agent_run_events(credentials, run_id)

    @staticmethod
    async def describe_failure(credentials: APIKeyCredentials, run: AgentRun) -> str:
        return await agent_api.describe_failure(credentials, run)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        run = parse_run(await self.get_run(credentials, input_data.research_id))
        task = ResearchTaskModel.from_agent_run(run)
        yield "research_id", task.research_id
        yield "status", task.status
        yield "instructions", task.instructions
        yield "model", task.model
        yield "created_at", task.created_at
        yield "finished_at", task.finished_at
        await bill_finished_run(self, run, user_id)
        for name, value in legacy_finished_outputs(run):
            yield name, value
        if task.cost_dollars:
            yield "cost_searches", task.cost_dollars.num_searches
            # Still emitted so saved workflows wired to them keep running.
            yield "cost_pages", task.cost_dollars.num_pages
            yield "cost_reasoning_tokens", task.cost_dollars.reasoning_tokens
        yield "error_message", await self._error_message(credentials, run)
        if input_data.include_events:
            events = await self.list_events(credentials, run.run_id)
            yield "events", [event.model_dump() for event in events]

    async def _error_message(
        self, credentials: APIKeyCredentials, run: AgentRun
    ) -> str | None:
        if run.status != "failed":
            return None
        return await self.describe_failure(credentials, run)


class ExaWaitForResearchBlock(Block):
    """Wait for a research task (an Exa Agent run) to complete."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        research_id: str = SchemaField(
            description="The ID of the research task to wait for (an Exa Agent run ID)",
            placeholder="agent_run_01jszdfs0052sg4jc552sg4jc5",
        )
        timeout: int = SchemaField(
            default=600,
            description="Maximum time to wait in seconds",
            ge=1,
            le=3600,
        )
        check_interval: int = SchemaField(
            default=10,
            description="Seconds between status checks",
            advanced=True,
            ge=1,
            le=60,
        )

    class Output(BlockSchemaOutput):
        research_id: str = SchemaField(description="The research task identifier")
        final_status: str = SchemaField(description="Final status when polling stopped")
        output_content: str | None = SchemaField(
            description="Research output as text (if completed)"
        )
        output_parsed: dict | None = SchemaField(
            description="Structured JSON output (if outputSchema was provided and completed)"
        )
        cost_total: float | None = SchemaField(description="Total cost in USD")
        elapsed_time: float = SchemaField(description="Total time waited in seconds")
        timed_out: bool = SchemaField(
            description="Whether polling timed out before completion"
        )

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-7890-4abc-9012-3456789abcde",
            description="Wait for a research task to complete with configurable "
            "timeout. Legacy: now waits on Exa Agent runs; use 'Wait for Exa Agent "
            "Run' for new workflows.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWaitForResearchBlock.Input,
            output_schema=ExaWaitForResearchBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "research_id": SAMPLE_RUN_ID,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("research_id", SAMPLE_RUN_ID),
                ("final_status", "completed"),
                ("elapsed_time", 1.0),
                ("timed_out", False),
                ("output_content", SAMPLE_COMPLETED_RUN["output"]["text"]),
                ("output_parsed", SAMPLE_COMPLETED_RUN["output"]["structured"]),
                ("cost_total", 0.1),
            ],
            test_mock={
                "wait_for_run": lambda *args, **kwargs: AgentWaitResult(
                    run=parse_run(SAMPLE_COMPLETED_RUN),
                    timed_out=False,
                    elapsed_seconds=1.0,
                )
            },
        )

    @staticmethod
    async def wait_for_run(
        credentials: APIKeyCredentials,
        run_id: str,
        timeout_seconds: float,
        interval_seconds: float,
    ) -> AgentWaitResult:
        return await agent_api.wait_for_agent_run(
            credentials,
            run_id,
            timeout_seconds,
            interval_seconds=interval_seconds,
            max_interval_seconds=interval_seconds,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        result = await self.wait_for_run(
            credentials,
            input_data.research_id,
            input_data.timeout,
            input_data.check_interval,
        )
        yield "research_id", result.run.run_id or input_data.research_id
        yield "final_status", legacy_status(result.run.status)
        yield "elapsed_time", result.elapsed_seconds
        yield "timed_out", result.timed_out
        if result.timed_out:
            return
        await bill_finished_run(self, result.run, user_id)
        for name, value in legacy_finished_outputs(result.run):
            yield name, value
