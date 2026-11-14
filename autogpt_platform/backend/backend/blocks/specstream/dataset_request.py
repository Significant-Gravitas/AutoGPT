"""Read a dataset request written in plain words and price it. Spends nothing."""

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
from backend.util.exceptions import BlockExecutionError, BlockInputError

from ._api import SpecStreamClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, specstream
from ._models import Quote, Reading
from ._testdata import TEST_QUOTE, TEST_READING, TEST_REQUEST

MAX_ROW_COUNT = 10_000


class SpecStreamCreateDatasetRequestBlock(Block):
    """Compile a request into a reading and a signed price. Nothing runs yet."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = specstream.credentials_field(
            description="The SpecStream integration requires an API Key."
        )
        request: str = SchemaField(
            description=(
                "The dataset to build, in plain words: what each row is, the "
                "conditions every row must meet, and the details to bring back."
            ),
            placeholder="Seed-stage fintech startups in France with website and founders",
            min_length=1,
            max_length=2000,
        )
        row_count: int = SchemaField(
            description=(
                "How many rows to ask for when the request names no number. "
                "A number in the request, such as '100 fintech startups', wins. "
                "The price is set per matching row."
            ),
            default=25,
            ge=1,
            le=MAX_ROW_COUNT,
        )

    class Output(BlockSchemaOutput):
        reading: str = SchemaField(
            description="How SpecStream read the request, as one sentence"
        )
        notices: list[str] = SchemaField(
            description="Notes to show a person before the quote is approved"
        )
        quote_id: str = SchemaField(description="The ID of the signed quote")
        quote_hash: str = SchemaField(
            description="The signature of the quote, needed to approve it"
        )
        maximum_cents: int = SchemaField(
            description="The most this run can cost, in cents. It is never exceeded."
        )
        currency: str = SchemaField(description="The currency of the quote")
        expires_at: str = SchemaField(description="When the quote stops being valid")
        price_summary: str = SchemaField(
            description="The quote in one sentence, for a person to review"
        )

    def __init__(self):
        super().__init__(
            id="4e565bf5-7d96-4126-9f9d-ee7676f08f6d",
            description=(
                "Reads a dataset request in plain words and returns a signed quote "
                "with the most it can cost. Nothing runs and nothing is charged."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            display_name="SpecStream Create Dataset Request",
            input_schema=SpecStreamCreateDatasetRequestBlock.Input,
            output_schema=SpecStreamCreateDatasetRequestBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "request": TEST_REQUEST,
                "row_count": 25,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("reading", "Fintech startups in France at seed stage"),
                ("notices", []),
                ("quote_id", "qte_test_001"),
                ("quote_hash", "b" * 64),
                ("maximum_cents", 375),
                ("currency", "USD"),
                ("expires_at", "2026-11-03T10:00:00+00:00"),
                ("price_summary", str),
            ],
            test_mock={
                "read_request": lambda *args, **kwargs: TEST_READING,
                "price_reading": lambda *args, **kwargs: TEST_QUOTE,
            },
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    @staticmethod
    async def read_request(credentials: APIKeyCredentials, text: str) -> Reading:
        return await SpecStreamClient(credentials).compile(text)

    @staticmethod
    async def price_reading(
        credentials: APIKeyCredentials, ast_hash: str, row_count: int
    ) -> Quote:
        return await SpecStreamClient(credentials).quote(ast_hash, row_count)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        refuse_platform_credentials(self, credentials)
        reading = await self.read_request(credentials, input_data.request)
        ast_hash = self._runnable_hash(reading)
        row_count = min(reading.stated_count or input_data.row_count, MAX_ROW_COUNT)
        quote = await self.price_reading(credentials, ast_hash, row_count)

        yield "reading", reading.canonical_sentence or ""
        yield "notices", [reading.notice] if reading.notice else []
        yield "quote_id", quote.quote_id
        yield "quote_hash", quote.quote_hash
        yield "maximum_cents", quote.maximum_cents
        yield "currency", quote.currency
        yield "expires_at", quote.expires_at.isoformat()
        yield "price_summary", price_summary(quote, row_count)

    def _runnable_hash(self, reading: Reading) -> str:
        if reading.is_runnable and reading.ast_hash:
            return reading.ast_hash
        if reading.status == "compiler_unavailable":
            raise BlockExecutionError(
                message=(
                    "SpecStream cannot read requests right now because the service "
                    "is down. Try again later."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        raise BlockInputError(
            message=unrunnable_message(reading),
            block_name=self.name,
            block_id=self.id,
        )


def price_summary(quote: Quote, row_count: int) -> str:
    amount = f"{quote.maximum_cents / 100:.2f} {quote.currency}"
    return (
        f"Up to {amount} for up to {row_count} rows. Only matching rows are "
        f"charged and the quote expires at {quote.expires_at.isoformat()}."
    )


def unrunnable_message(reading: Reading) -> str:
    reasons = [item.reason for item in reading.unsupported if item.reason]
    detail = f" {' '.join(reasons)}" if reasons else ""
    sentence: str | None = reading.canonical_sentence
    read_as = f" It was read as: {sentence}." if sentence else ""
    return (
        f"SpecStream could not turn this request into a runnable dataset "
        f"(status: {reading.status}).{read_as}{detail} Reword the request and try again."
    )
