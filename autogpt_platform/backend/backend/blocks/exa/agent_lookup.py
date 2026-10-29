"""
Exa Agent blocks that read a run: get its current state, or wait for it.

Both report a failed run through ``failure_reason`` rather than failing the
block, so a workflow can branch on ``status``.
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    CredentialsMetaInput,
    SchemaField,
)

from . import agent_api
from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .agent import RUN_ID_PLACEHOLDER, AgentRunOutputs, run_outputs
from .agent_api import AgentWaitResult
from .agent_billing import bill_finished_run
from .agent_fixtures import SAMPLE_COMPLETED_RUN, SAMPLE_RUN_ID
from .agent_models import AgentRun, parse_run

FAILURE_REASON_DESCRIPTION = "Why the run failed (only when status is failed)"


class ExaGetAgentRunBlock(Block):
    """Fetch an Exa Agent run's current status and, once finished, its result."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        run_id: str = SchemaField(
            description="ID of the Exa Agent run", placeholder=RUN_ID_PLACEHOLDER
        )

    class Output(AgentRunOutputs):
        query: str = SchemaField(description="The query the run was given")
        effort: str | None = SchemaField(description="The effort the run used")
        failure_reason: str = SchemaField(description=FAILURE_REASON_DESCRIPTION)

    def __init__(self):
        super().__init__(
            id="62dd0198-36aa-4586-bbd1-83318c185ac6",
            description="Get an Exa Agent run's status and, once it has finished, "
            "its answer, structured output, citations and cost",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetAgentRunBlock.Input,
            output_schema=ExaGetAgentRunBlock.Output,
            effect=BlockEffect.READ,
            display_name="Get Exa Agent Run",
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                *run_outputs(parse_run(SAMPLE_COMPLETED_RUN)),
                ("query", SAMPLE_COMPLETED_RUN["request"]["query"]),
                ("effort", "medium"),
            ],
            test_mock={"get_run": lambda *args, **kwargs: SAMPLE_COMPLETED_RUN},
        )

    @staticmethod
    async def get_run(credentials: APIKeyCredentials, run_id: str) -> dict:
        return await agent_api.get_agent_run(credentials, run_id)

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
        run = parse_run(await self.get_run(credentials, input_data.run_id))
        await bill_finished_run(self, run, user_id)
        for name, value in run_outputs(run):
            yield name, value
        yield "query", run.query
        yield "effort", run.effort
        if run.status == "failed":
            yield "failure_reason", await self.describe_failure(credentials, run)


class ExaWaitForAgentRunBlock(Block):
    """Poll an Exa Agent run until it finishes or the timeout passes."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        run_id: str = SchemaField(
            description="ID of the Exa Agent run to wait for",
            placeholder=RUN_ID_PLACEHOLDER,
        )
        timeout: int = SchemaField(
            default=900,
            description="Longest time to wait in seconds (1-10800). Ultra runs "
            "usually take about 30 minutes and can take up to 3 hours.",
            ge=1,
            le=10800,
            advanced=False,
        )
        check_interval: int = SchemaField(
            default=5,
            description="Seconds before the first re-check (1-60); the gap then "
            "grows gradually up to 30 seconds",
            ge=1,
            le=60,
        )

    class Output(AgentRunOutputs):
        timed_out: bool = SchemaField(
            description="True if the run was still going when the timeout passed"
        )
        elapsed_seconds: float = SchemaField(description="How long the block waited")
        failure_reason: str = SchemaField(description=FAILURE_REASON_DESCRIPTION)

    def __init__(self):
        super().__init__(
            id="33a4c12a-b81b-4a95-90b1-ba2ebc52299d",
            description="Wait for an Exa Agent run to finish, then return its "
            "answer, structured output, citations and cost",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWaitForAgentRunBlock.Input,
            output_schema=ExaWaitForAgentRunBlock.Output,
            effect=BlockEffect.READ,
            display_name="Wait for Exa Agent Run",
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "run_id": SAMPLE_RUN_ID},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                *run_outputs(parse_run(SAMPLE_COMPLETED_RUN)),
                ("timed_out", False),
                ("elapsed_seconds", 1.0),
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
            credentials, run_id, timeout_seconds, interval_seconds
        )

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
        result = await self.wait_for_run(
            credentials,
            input_data.run_id,
            input_data.timeout,
            input_data.check_interval,
        )
        await bill_finished_run(self, result.run, user_id)
        for name, value in run_outputs(result.run):
            yield name, value
        yield "timed_out", result.timed_out
        yield "elapsed_seconds", result.elapsed_seconds
        if result.run.status == "failed":
            yield "failure_reason", await self.describe_failure(credentials, result.run)
