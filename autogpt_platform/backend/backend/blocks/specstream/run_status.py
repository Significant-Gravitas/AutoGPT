"""Read a run's status, or wait for it to finish by polling."""

from asyncio import sleep
from time import monotonic

from backend.blocks._platform_credentials import refuse_platform_credentials
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

from ._api import SpecStreamClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, specstream
from ._models import RunStatus
from ._testdata import TEST_COMPLETED, TEST_FIND_ID, TEST_RUNNING

BACKOFF_FACTOR = 1.5
MAX_WAIT_SECONDS = 3600


class SpecStreamGetRunStatusBlock(Block):
    """Read where a dataset run stands. Free, and safe to repeat."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = specstream.credentials_field(
            description="The SpecStream integration requires an API Key."
        )
        run_id: str = SchemaField(
            description="The run ID from Approve and Run", placeholder="fnd_..."
        )

    class Output(BlockSchemaOutput):
        run_id: str = SchemaField(description="The ID of the run")
        status: str = SchemaField(
            description="queued, planning, running, completed, partial, failed, canceled or expired"
        )
        is_done: bool = SchemaField(description="True once the run has ended")
        requested_rows: int = SchemaField(description="How many rows were asked for")
        accepted_rows: int = SchemaField(
            description="How many rows met every condition so far"
        )
        charged_cents: int = SchemaField(
            description="What the run has charged so far, in cents"
        )
        stop_reason: str = SchemaField(
            description="Why the run ended, empty while it runs"
        )

    def __init__(self):
        super().__init__(
            id="d5cf71bd-7f29-4433-8189-920eb61b2d5b",
            description="Reads the status of a SpecStream dataset run. Costs nothing.",
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            display_name="SpecStream Get Run Status",
            input_schema=SpecStreamGetRunStatusBlock.Input,
            output_schema=SpecStreamGetRunStatusBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "run_id": TEST_FIND_ID},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("run_id", TEST_FIND_ID),
                ("status", "running"),
                ("is_done", False),
                ("requested_rows", 25),
                ("accepted_rows", 7),
                ("charged_cents", 0),
                ("stop_reason", ""),
            ],
            test_mock={"get_run": lambda *args, **kwargs: TEST_RUNNING},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def get_run(credentials: APIKeyCredentials, run_id: str) -> RunStatus:
        return await SpecStreamClient(credentials).get_run(run_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        refuse_platform_credentials(self, credentials)
        run = await self.get_run(credentials, input_data.run_id)
        yield "run_id", run.find_id
        yield "status", run.status
        yield "is_done", run.is_done
        yield "requested_rows", run.requested_count
        yield "accepted_rows", run.accepted_count
        yield "charged_cents", charged_cents(run)
        yield "stop_reason", run.stop_reason or ""


class SpecStreamWaitForRunBlock(Block):
    """Poll a run until it ends or the timeout passes. No webhook is needed."""

    # The default 30-minute block limit would end the wait before the longest
    # timeout, so the block reports timed_out itself.
    execution_timeout_seconds: int | None = MAX_WAIT_SECONDS + 300

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = specstream.credentials_field(
            description="The SpecStream integration requires an API Key."
        )
        run_id: str = SchemaField(
            description="The run ID from Approve and Run", placeholder="fnd_..."
        )
        timeout: int = SchemaField(
            description=(
                "The longest to wait, in seconds. A run takes minutes, and a large "
                "one more. If this passes, the run keeps going and can be read again."
            ),
            default=900,
            ge=1,
            le=MAX_WAIT_SECONDS,
        )
        check_interval: int = SchemaField(
            description="Seconds between the first status checks",
            default=10,
            ge=1,
            le=60,
            advanced=True,
        )
        max_interval: int = SchemaField(
            description="The longest gap between checks, as the gap grows",
            default=60,
            ge=1,
            le=300,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        run_id: str = SchemaField(description="The ID of the run")
        status: str = SchemaField(description="The last status read")
        is_done: bool = SchemaField(description="True if the run ended in time")
        timed_out: bool = SchemaField(description="True if the timeout passed first")
        elapsed_seconds: float = SchemaField(description="How long the block waited")
        accepted_rows: int = SchemaField(description="How many rows have matched")
        charged_cents: int = SchemaField(description="What the run charged, in cents")

    def __init__(self):
        super().__init__(
            id="a816eec6-053c-4919-9b46-bccbb4a562f5",
            description=(
                "Waits for a SpecStream dataset run to end by checking its status. "
                "Costs nothing."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA, BlockCategory.LOGIC},
            display_name="SpecStream Wait For Run",
            input_schema=SpecStreamWaitForRunBlock.Input,
            output_schema=SpecStreamWaitForRunBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "run_id": TEST_FIND_ID},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("run_id", TEST_FIND_ID),
                ("status", "completed"),
                ("is_done", True),
                ("timed_out", False),
                ("elapsed_seconds", float),
                ("accepted_rows", 25),
                ("charged_cents", 375),
            ],
            test_mock={"get_run": lambda *args, **kwargs: TEST_COMPLETED},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def get_run(credentials: APIKeyCredentials, run_id: str) -> RunStatus:
        return await SpecStreamClient(credentials).get_run(run_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        refuse_platform_credentials(self, credentials)
        started = monotonic()
        gap = float(input_data.check_interval)
        run = await self.get_run(credentials, input_data.run_id)
        while not run.is_done and monotonic() - started < input_data.timeout:
            await sleep(min(gap, max(input_data.timeout - (monotonic() - started), 0)))
            gap = min(gap * BACKOFF_FACTOR, float(input_data.max_interval))
            run = await self.get_run(credentials, input_data.run_id)

        yield "run_id", run.find_id
        yield "status", run.status
        yield "is_done", run.is_done
        yield "timed_out", not run.is_done
        yield "elapsed_seconds", monotonic() - started
        yield "accepted_rows", run.accepted_count
        yield "charged_cents", charged_cents(run)


def charged_cents(run: RunStatus) -> int:
    return run.commerce.amount_captured_cents if run.commerce else 0
