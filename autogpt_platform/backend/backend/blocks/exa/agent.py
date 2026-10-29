"""
Exa Agent blocks: start a run, look one up, and wait for one to finish.

Exa Agent is Exa's asynchronous research agent (``/agent/runs``). It plans and
runs many searches, reads pages, verifies findings and returns a grounded
answer, optionally as JSON matching your schema.
"""

from collections.abc import Iterator
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

from . import agent_api
from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .agent_api import AgentRunRequest, AgentWaitResult, build_create_payload
from .agent_billing import bill_finished_run
from .agent_fixtures import SAMPLE_COMPLETED_RUN, SAMPLE_QUEUED_RUN
from .agent_models import (
    DATA_SOURCES_DESCRIPTION,
    EFFORT_DESCRIPTION,
    AgentDataSource,
    AgentEffort,
    AgentGrounding,
    AgentRun,
    AgentUsage,
    parse_run,
)

RUN_ID_PLACEHOLDER = "agent_run_01jszdfs0052sg4jc552sg4jc5"


class AgentRunOutputs(BlockSchemaOutput):
    """Outputs every block that returns a whole run shares."""

    run_id: str = SchemaField(description="Exa Agent run ID (starts with agent_run_)")
    status: str = SchemaField(
        description="queued, running, completed, failed or cancelled"
    )
    stop_reason: str | None = SchemaField(
        description="Why the run stopped: schema_satisfied (done), budget_reached, "
        "time_limit_reached, iteration_limit_reached, token_limit_reached, "
        "stopped (you stopped it; partial results kept), error or cancelled"
    )
    text: str = SchemaField(description="The natural-language answer")
    structured: Any | None = SchemaField(
        description="JSON matching your output schema (empty without a schema). "
        "Fields Exa could not support with evidence may be null."
    )
    grounding: list[AgentGrounding] = SchemaField(
        description="Citations for each output field, with confidence"
    )
    usage: AgentUsage = SchemaField(
        title="Usage",
        description="Billed usage: compute units, searches, enrichments, data source calls",
    )
    cost_total: float = SchemaField(description="Total cost of the run in USD")
    created_at: str = SchemaField(description="When the run was created (ISO 8601)")
    completed_at: str | None = SchemaField(
        description="When the run finished (ISO 8601), empty while running"
    )


class ExaAgentRunBlock(Block):
    """Start an Exa Agent run and (by default) wait for its result."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        query: str = SchemaField(
            description="What you want researched, built or enriched, in plain "
            "language. Be specific about criteria and how many results you want.",
            placeholder="Find 10 AI infrastructure companies that raised a Series A "
            "in the last 6 months, with their founders and websites",
        )
        effort: AgentEffort = SchemaField(
            title="Effort",
            default=AgentEffort.AUTO,
            description=EFFORT_DESCRIPTION,
            advanced=False,
        )
        output_schema: dict | None = SchemaField(
            default=None,
            description="JSON Schema for the answer. When set, the result is "
            "returned as validated JSON in 'structured'. Make fields that may not "
            "be verifiable nullable, and bound lists with maxItems.",
            advanced=False,
        )
        wait_for_completion: bool = SchemaField(
            default=True,
            description="Wait for the run to finish and return its answer. Turn "
            "off to get the run ID immediately and collect it later with "
            "'Wait for Exa Agent Run' or 'Get Exa Agent Run'.",
            advanced=False,
        )
        system_prompt: str | None = SchemaField(
            default=None,
            description="Extra instructions for how to research or answer, such "
            "as preferred sources or rules for what counts as a match",
            placeholder="Only use official company websites as evidence.",
        )
        input_data: list[dict[str, Any]] = SchemaField(
            default_factory=list,
            description="Records for the run to process or enrich, one object per "
            'record, e.g. [{"company": "Acme", "domain": "acme.com"}]',
        )
        exclusions: list[dict[str, Any]] = SchemaField(
            default_factory=list,
            description="Records or entities the run must not return, e.g. ones "
            'you already have: [{"company": "Acme"}]',
        )
        previous_run_id: str | None = SchemaField(
            default=None,
            description="ID of an earlier run to follow up on (e.g. 'find 10 "
            "more'). The follow-up gets its own new run ID.",
            placeholder=RUN_ID_PLACEHOLDER,
        )
        data_sources: list[AgentDataSource] = SchemaField(
            default_factory=list,
            description=DATA_SOURCES_DESCRIPTION,
        )
        custom_data_sources: list[str] = SchemaField(
            default_factory=list,
            description="Other Exa Connect provider IDs that Exa enabled for your "
            "team on request (e.g. harmonic). Counts toward the limit of 5.",
        )
        max_cost_dollars: float | None = SchemaField(
            default=None,
            description="Spending cap for this run in USD ($1-$100). Only for the "
            "auto and ultra efforts; Exa's default cap is $5 for auto and $20 "
            "for ultra.",
            ge=1,
            le=100,
        )
        max_duration_seconds: int | None = SchemaField(
            default=None,
            description="Soft time limit in seconds (300-10800, i.e. 5 minutes to "
            "3 hours). Ultra effort only: near the limit the run wraps up and "
            "returns what it found.",
            ge=300,
            le=10800,
        )
        metadata: dict[str, str] = SchemaField(
            default_factory=dict,
            description="Your own key-value labels stored with the run",
        )
        user_identifier: str | None = SchemaField(
            default=None,
            description="Opaque ID of the end user this run is for (max 64 "
            "characters), so Exa can attribute usage to them",
            max_length=64,
        )
        polling_timeout: int = SchemaField(
            default=900,
            description="How long to wait for the run in seconds (10-10800). If it "
            "is still going, the block fails with the run ID so you can collect "
            "it later; the run keeps going on Exa.",
            ge=10,
            le=10800,
        )

    class Output(AgentRunOutputs):
        pass

    def __init__(self):
        super().__init__(
            id="9f3ddfa5-2ccb-4b49-a745-8244d9e32422",
            description="Run Exa Agent, Exa's research agent, for deep research, "
            "list building and enrichment. Returns a cited answer, optionally as "
            "JSON matching your schema.",
            categories={BlockCategory.SEARCH, BlockCategory.AI},
            input_schema=ExaAgentRunBlock.Input,
            output_schema=ExaAgentRunBlock.Output,
            effect=BlockEffect.READ,
            display_name="Exa Agent",
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "Find the top 3 AI coding assistants and their pricing",
                "effort": "medium",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                *_expected_outputs(SAMPLE_COMPLETED_RUN),
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
        return await agent_api.wait_for_agent_run(credentials, run_id, timeout_seconds)

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
        payload = build_create_payload(_to_request(input_data))
        created = parse_run(await self.create_run(credentials, payload))
        if not input_data.wait_for_completion:
            yield "run_id", created.run_id
            yield "status", created.status
            yield "created_at", created.created_at
            return

        result = await self.wait_for_run(
            credentials, created.run_id, input_data.polling_timeout
        )
        run = result.run
        if result.timed_out:
            yield "run_id", run.run_id
            yield "status", run.status
            yield "created_at", run.created_at
            yield "error", _timeout_message(run, input_data.polling_timeout)
            return

        await bill_finished_run(self, run, user_id)
        for name, value in run_outputs(run):
            yield name, value
        if run.status == "failed":
            yield "error", await self.describe_failure(credentials, run)


def run_outputs(run: AgentRun) -> Iterator[tuple[str, Any]]:
    """Yield the shared :class:`AgentRunOutputs` fields for ``run``."""
    yield "run_id", run.run_id
    yield "status", run.status
    yield "stop_reason", run.stop_reason
    yield "text", run.text
    yield "structured", run.structured
    yield "grounding", run.grounding
    yield "usage", run.usage
    yield "cost_total", run.cost.total
    yield "created_at", run.created_at
    yield "completed_at", run.completed_at


def _expected_outputs(raw_run: dict) -> list[tuple[str, Any]]:
    return list(run_outputs(parse_run(raw_run)))


def _to_request(input_data: ExaAgentRunBlock.Input) -> AgentRunRequest:
    return AgentRunRequest(
        query=input_data.query,
        effort=input_data.effort,
        system_prompt=input_data.system_prompt,
        input_data=input_data.input_data,
        exclusions=input_data.exclusions,
        output_schema=input_data.output_schema,
        previous_run_id=input_data.previous_run_id,
        metadata=input_data.metadata,
        data_sources=input_data.data_sources,
        custom_data_sources=input_data.custom_data_sources,
        max_cost_dollars=input_data.max_cost_dollars,
        max_duration_seconds=input_data.max_duration_seconds,
        user_identifier=input_data.user_identifier,
    )


def _timeout_message(run: AgentRun, timeout_seconds: int) -> str:
    return (
        f"Exa Agent run {run.run_id} was still {run.status} after "
        f"{timeout_seconds} seconds. It keeps running on Exa: collect it with "
        "'Wait for Exa Agent Run' or 'Get Exa Agent Run' using this run ID, or "
        "raise the polling timeout."
    )
