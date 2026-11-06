"""Approve a quote and start the run. This is the only block that spends."""

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
from backend.util.exceptions import BlockExecutionError

from ._api import SpecStreamClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, specstream
from ._models import Execution
from ._testdata import TEST_EXECUTION, TEST_QUOTE_HASH

KEY_PREFIX = "autogpt-"
MIN_KEY_LENGTH = 8
MAX_KEY_LENGTH = 128


class SpecStreamApproveAndRunBlock(Block):
    """Approve one quote at a stated maximum and start the dataset run."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = specstream.credentials_field(
            description="The SpecStream integration requires an API Key."
        )
        quote_id: str = SchemaField(
            description="The quote ID from Create Dataset Request",
            placeholder="qte_...",
        )
        quote_hash: str = SchemaField(
            description="The quote signature from Create Dataset Request",
            min_length=64,
            max_length=64,
        )
        confirm_max_cents: int = SchemaField(
            description=(
                "The most you approve this run to cost, in cents. It must equal "
                "the maximum_cents of the quote, or nothing runs. Only matching "
                "rows are charged, and what the run does not use goes back to "
                "your balance."
            ),
            ge=1,
        )
        idempotency_key: str | None = SchemaField(
            description=(
                "A stable key so a retry never starts a second run. Defaults to "
                "one made from the quote ID, and a quote starts one run at most."
            ),
            default=None,
            advanced=True,
            min_length=MIN_KEY_LENGTH,
            max_length=MAX_KEY_LENGTH,
        )

    class Output(BlockSchemaOutput):
        run_id: str = SchemaField(
            description="The ID of the run, for Get Run Status and List Rows"
        )
        state: str = SchemaField(
            description="The state of the approved work: reserved, running, committed or released"
        )
        reserved_cents: int = SchemaField(
            description="The maximum held from your balance for this run, in cents"
        )
        replay: bool = SchemaField(
            description="True when this key had already approved this quote"
        )

    def __init__(self):
        super().__init__(
            id="eeb3013b-016a-49f7-9fe1-88a0ad955d66",
            description=(
                "Approves a SpecStream quote at the stated maximum and starts the "
                "dataset run. The run takes minutes and is charged per matching row."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            display_name="SpecStream Approve And Run",
            input_schema=SpecStreamApproveAndRunBlock.Input,
            output_schema=SpecStreamApproveAndRunBlock.Output,
            is_irreversible_action=True,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "quote_id": "qte_test_001",
                "quote_hash": TEST_QUOTE_HASH,
                "confirm_max_cents": 375,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("run_id", "fnd_test_001"),
                ("state", "running"),
                ("reserved_cents", 375),
                ("replay", False),
            ],
            test_mock={"authorize": lambda *args, **kwargs: TEST_EXECUTION},
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def authorize(
        credentials: APIKeyCredentials, input_data: Input, key: str
    ) -> Execution:
        client = SpecStreamClient(credentials)
        execution = await client.authorize(
            input_data.quote_id,
            input_data.quote_hash,
            input_data.confirm_max_cents,
            key,
        )
        if execution.find_id:
            return execution
        quote = await client.get_quote(input_data.quote_id)
        found = quote.existing_execution.find_id if quote.existing_execution else None
        return execution.model_copy(update={"find_id": found})

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        refuse_platform_credentials(self, credentials)
        key = input_data.idempotency_key or request_key(input_data.quote_id)
        execution = await self.authorize(credentials, input_data, key)
        if not execution.find_id:
            raise BlockExecutionError(
                message=(
                    f"SpecStream approved the quote (reservation "
                    f"{execution.reservation_id}) but has not returned a run ID yet. "
                    "Approve the same quote again to read it. A quote starts one run at most."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        yield "run_id", execution.find_id
        yield "state", execution.state
        yield "reserved_cents", execution.maximum_cents
        yield "replay", execution.replay


def request_key(quote_id: str) -> str:
    return f"{KEY_PREFIX}{quote_id}"[:MAX_KEY_LENGTH]
