"""Blocks that hand a design to a person on rmfg.com and read back their choices."""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import RMFGClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._inputs import credentials_field
from ._models_commerce import ReviewLink
from ._testdata import TEST_CONFIGURATION, TEST_REVIEW_LINK
from ._types import ManufacturingConfiguration

CATEGORIES = {BlockCategory.HARDWARE, BlockCategory.DATA}


class RMFGReviewLinkOutput(BlockSchemaOutput):
    review_link: ReviewLink = SchemaField(description="The review link resource")
    link_id: str = SchemaField(description="Link ID, to read the saved result back")
    review_url: str = SchemaField(
        description="Website page for a person to inspect and configure the design; keep private"
    )
    configuration: ManufacturingConfiguration = SchemaField(
        description="The configuration on the link, including any saved changes"
    )
    configuration_updated_at: str = SchemaField(
        description="When a person last saved changes; empty until they do"
    )
    status: str = SchemaField(description="open or expired")
    error: str = SchemaField(description="Error message if the request failed")


async def emit_link(link: ReviewLink) -> BlockOutput:
    yield "review_link", link
    yield "link_id", link.id
    yield "review_url", link.review_url
    yield "configuration", link.configuration
    yield "configuration_updated_at", link.configuration_updated_at or ""
    yield "status", link.status


LINK_TEST_OUTPUT = [
    ("review_link", TEST_REVIEW_LINK),
    ("link_id", TEST_REVIEW_LINK.id),
    ("review_url", TEST_REVIEW_LINK.review_url),
    ("configuration", TEST_CONFIGURATION),
    ("configuration_updated_at", ""),
    ("status", "open"),
]


class RMFGCreateReviewLinkBlock(Block):
    """Mint a website link where a person can adjust a configuration and save it."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        design_id: str = SchemaField(description="Design ID from Analyze Design")
        configuration: ManufacturingConfiguration = SchemaField(
            description="Starting configuration shown on the page; may be empty.",
            default_factory=ManufacturingConfiguration,
            advanced=False,
        )
        dfm_id: str = SchemaField(
            description="Attach a DFM report so the page opens that report's exact configuration.",
            default="",
            advanced=True,
        )
        idempotency_key: str = SchemaField(
            description="Stable key for identical retries; defaults to the node execution ID.",
            default="",
            advanced=True,
        )

    class Output(RMFGReviewLinkOutput):
        pass

    def __init__(self):
        super().__init__(
            id="9409dd93-3ce2-4fef-b14d-b70b1ecf6b17",
            description="Creates an RMFG review link so a person can inspect and adjust a design",
            categories=CATEGORIES,
            input_schema=RMFGCreateReviewLinkBlock.Input,
            output_schema=RMFGCreateReviewLinkBlock.Output,
            test_input={
                "design_id": TEST_REVIEW_LINK.design_id,
                "configuration": TEST_CONFIGURATION.model_dump(),
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=LINK_TEST_OUTPUT,
            test_mock={"create_link": lambda *args, **kwargs: TEST_REVIEW_LINK},
        )

    @staticmethod
    async def create_link(
        credentials: APIKeyCredentials, input_data: Input, idempotency_key: str
    ) -> ReviewLink:
        return await RMFGClient(credentials).create_review_link(
            input_data.design_id,
            input_data.configuration,
            input_data.dfm_id,
            idempotency_key,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        node_exec_id: str = "",
        **kwargs,
    ) -> BlockOutput:
        link = await self.create_link(
            credentials, input_data, input_data.idempotency_key or node_exec_id
        )
        async for output in emit_link(link):
            yield output


class RMFGGetReviewLinkBlock(Block):
    """Read a review link back, including the configuration a person saved."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        link_id: str = SchemaField(
            description="Link ID from Create Review Link, or a design's review_link_id"
        )

    class Output(RMFGReviewLinkOutput):
        pass

    def __init__(self):
        super().__init__(
            id="d82a0bed-6e69-48ae-b391-3eb84842069e",
            description="Fetches an RMFG review link and the configuration a person saved on it",
            categories=CATEGORIES,
            input_schema=RMFGGetReviewLinkBlock.Input,
            output_schema=RMFGGetReviewLinkBlock.Output,
            test_input={
                "link_id": TEST_REVIEW_LINK.id,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=LINK_TEST_OUTPUT,
            test_mock={"get_link": lambda *args, **kwargs: TEST_REVIEW_LINK},
        )

    @staticmethod
    async def get_link(credentials: APIKeyCredentials, link_id: str) -> ReviewLink:
        return await RMFGClient(credentials).get_review_link(link_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        link = await self.get_link(credentials, input_data.link_id)
        async for output in emit_link(link):
            yield output
