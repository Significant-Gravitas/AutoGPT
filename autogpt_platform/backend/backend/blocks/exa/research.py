"""
Exa Research blocks (legacy), now running on Exa Agent.

Exa retired the Research API (``/research/v1`` returns 410 RESEARCH_RETIRED).
These blocks keep their IDs, inputs and outputs so saved workflows keep
working, and send their work to ``/agent/runs`` instead:

- model -> effort: exa-research-fast -> low, exa-research -> medium,
  exa-research-pro -> high
- instructions -> query, output_schema -> outputSchema
- research_id is the Exa Agent run ID; statuses are mapped back to
  pending / running / completed / canceled / failed

New workflows should use the Exa Agent blocks ("Exa Agent", "Get Exa Agent
Run", "Wait for Exa Agent Run", "List Exa Agent Runs"), which expose every
Agent option. See research_models.py for the fields Exa Agent no longer reports.
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
from ._platform_key import require_own_exa_key
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .agent_api import AgentRunRequest, AgentWaitResult, build_create_payload
from .agent_billing import bill_finished_run
from .agent_fixtures import (
    SAMPLE_COMPLETED_RUN,
    SAMPLE_QUEUED_RUN,
    SAMPLE_RUN_ID,
    SAMPLE_RUN_LIST,
)
from .agent_models import parse_page, parse_run
from .research_models import (
    MODEL_TO_EFFORT,
    ResearchModel,
    ResearchTaskModel,
    iso_to_ms,
    legacy_finished_outputs,
    legacy_status,
)
from .research_queries import ExaGetResearchBlock, ExaWaitForResearchBlock

__all__ = [
    "ExaCreateResearchBlock",
    "ExaGetResearchBlock",
    "ExaListResearchBlock",
    "ExaWaitForResearchBlock",
    "ResearchModel",
    "ResearchTaskModel",
]

LEGACY_POLL_INTERVAL_SECONDS = 10

_COMPLETED = ResearchTaskModel.from_agent_run(parse_run(SAMPLE_COMPLETED_RUN))


class ExaCreateResearchBlock(Block):
    """Create a research task (now an Exa Agent run) and optionally wait for it."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        instructions: str = SchemaField(
            description="Research instructions - clearly define what information to find, how to conduct research, and desired output format.",
            placeholder="Research the top 5 AI coding assistants, their features, pricing, and user reviews",
        )
        model: ResearchModel = SchemaField(
            default=ResearchModel.STANDARD,
            description="Research depth, now mapped to an Exa Agent effort: "
            "'exa-research-fast' runs at low effort ($0.025), 'exa-research' at "
            "medium ($0.10), 'exa-research-pro' at high ($0.50). The Exa Agent "
            "block offers every effort.",
        )
        output_schema: dict | None = SchemaField(
            default=None,
            description="JSON Schema to enforce structured output. When provided, results are validated and returned as parsed JSON.",
            advanced=True,
        )
        wait_for_completion: bool = SchemaField(
            default=True,
            description="Wait for research to complete before returning. Ensures you get results immediately.",
        )
        polling_timeout: int = SchemaField(
            default=600,
            description="Maximum time to wait for completion in seconds (only if wait_for_completion is True)",
            advanced=True,
            ge=1,
            le=3600,
        )

    class Output(BlockSchemaOutput):
        research_id: str = SchemaField(
            description="Unique identifier for tracking this research request (the Exa Agent run ID)"
        )
        status: str = SchemaField(description="Final status of the research")
        model: str = SchemaField(description="The research model used")
        instructions: str = SchemaField(
            description="The research instructions provided"
        )
        created_at: int = SchemaField(
            description="When the research was created (Unix timestamp in ms)"
        )
        output_content: str | None = SchemaField(
            description="Research output as text (only if wait_for_completion was True and completed)"
        )
        output_parsed: dict | None = SchemaField(
            description="Structured JSON output (only if wait_for_completion and outputSchema were provided)"
        )
        cost_total: float | None = SchemaField(
            description="Total cost in USD (only if wait_for_completion was True and completed)"
        )
        elapsed_time: float | None = SchemaField(
            description="Time taken to complete in seconds (only if wait_for_completion was True)"
        )

    def __init__(self):
        super().__init__(
            id="a1f2e3d4-c5b6-4a78-9012-3456789abcde",
            description="Create a research task with optional waiting. Legacy: now "
            "runs on Exa Agent (Exa retired its Research API); use the 'Exa Agent' "
            "block for new workflows.",
            categories={BlockCategory.SEARCH, BlockCategory.AI},
            input_schema=ExaCreateResearchBlock.Input,
            output_schema=ExaCreateResearchBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "instructions": "Find the top 3 AI coding assistants and their pricing",
                "model": ResearchModel.STANDARD.value,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("research_id", SAMPLE_RUN_ID),
                ("status", "completed"),
                ("model", "exa-research"),
                ("instructions", _COMPLETED.instructions),
                ("created_at", _COMPLETED.created_at),
                ("elapsed_time", 1.0),
                ("output_content", SAMPLE_COMPLETED_RUN["output"]["text"]),
                ("output_parsed", SAMPLE_COMPLETED_RUN["output"]["structured"]),
                ("cost_total", 0.1),
            ],
            test_mock={
                "create_run": lambda *args, **kwargs: SAMPLE_QUEUED_RUN,
                "wait_for_run": lambda *args, **kwargs: AgentWaitResult(
                    run=parse_run(SAMPLE_COMPLETED_RUN),
                    timed_out=False,
                    elapsed_seconds=1.0,
                ),
            },
        )

    @staticmethod
    async def create_run(credentials: APIKeyCredentials, payload: dict) -> dict:
        return await agent_api.create_agent_run(credentials, payload)

    @staticmethod
    async def wait_for_run(
        credentials: APIKeyCredentials, run_id: str, timeout_seconds: float
    ) -> AgentWaitResult:
        return await agent_api.wait_for_agent_run(
            credentials,
            run_id,
            timeout_seconds,
            interval_seconds=LEGACY_POLL_INTERVAL_SECONDS,
            max_interval_seconds=LEGACY_POLL_INTERVAL_SECONDS,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        payload = build_create_payload(
            AgentRunRequest(
                query=input_data.instructions,
                effort=MODEL_TO_EFFORT[input_data.model],
                output_schema=input_data.output_schema,
            )
        )
        created = parse_run(await self.create_run(credentials, payload))
        if not input_data.wait_for_completion:
            yield "research_id", created.run_id
            yield "status", legacy_status(created.status)
            yield "model", input_data.model.value
            yield "instructions", created.query or input_data.instructions
            yield "created_at", iso_to_ms(created.created_at) or 0
            return

        result = await self.wait_for_run(
            credentials, created.run_id, input_data.polling_timeout
        )
        if result.timed_out:
            raise ValueError(
                f"Research {created.run_id} did not complete within "
                f"{input_data.polling_timeout} seconds"
            )
        yield "research_id", result.run.run_id
        yield "status", legacy_status(result.run.status)
        yield "model", input_data.model.value
        yield "instructions", result.run.query or input_data.instructions
        yield "created_at", iso_to_ms(result.run.created_at) or 0
        yield "elapsed_time", result.elapsed_seconds
        await bill_finished_run(self, result.run, user_id)
        for name, value in legacy_finished_outputs(result.run):
            yield name, value


class ExaListResearchBlock(Block):
    """List research tasks (Exa Agent runs) with pagination support."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        cursor: str | None = SchemaField(
            default=None,
            description="Cursor for pagination through results",
            advanced=True,
        )
        limit: int = SchemaField(
            default=10,
            description="Number of research tasks to return (1-50)",
            ge=1,
            le=50,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        research_tasks: list[ResearchTaskModel] = SchemaField(
            description="List of research tasks ordered by creation time (newest first)"
        )
        research_task: ResearchTaskModel = SchemaField(
            description="Individual research task (yielded for each task)"
        )
        has_more: bool = SchemaField(
            description="Whether there are more tasks to paginate through"
        )
        next_cursor: str | None = SchemaField(
            description="Cursor for the next page of results"
        )

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-8901-4bcd-9012-3456789abcde",
            description="List research tasks with pagination support. Legacy: now "
            "lists all of your team's Exa Agent runs; use 'List Exa Agent Runs' for "
            "new workflows.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListResearchBlock.Input,
            output_schema=ExaListResearchBlock.Output,
            effect=BlockEffect.READ,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "limit": 1},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("research_tasks", [_COMPLETED]),
                ("research_task", _COMPLETED),
                ("has_more", True),
                ("next_cursor", SAMPLE_RUN_LIST["nextCursor"]),
            ],
            test_mock={"list_runs": lambda *args, **kwargs: SAMPLE_RUN_LIST},
        )

    @staticmethod
    async def list_runs(
        credentials: APIKeyCredentials, limit: int, cursor: str | None
    ) -> dict:
        return await agent_api.list_agent_runs(credentials, limit, cursor)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        require_own_exa_key(credentials, "Listing Exa research tasks")
        data = await self.list_runs(credentials, input_data.limit, input_data.cursor)
        tasks = [
            ResearchTaskModel.from_agent_run(parse_run(run))
            for run in data.get("data") or []
        ]
        page = parse_page(data)
        yield "research_tasks", tasks
        for task in tasks:
            yield "research_task", task
        yield "has_more", page.has_more
        yield "next_cursor", page.next_cursor
