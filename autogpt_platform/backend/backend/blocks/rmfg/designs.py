"""Blocks that upload a STEP file for analysis and read the resulting design."""

from pathlib import Path

from backend.data.execution import ExecutionContext
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    MediaFileType,
    SchemaField,
    store_media_file,
)
from backend.util.file import get_exec_file_path

from ._api import RMFGClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._inputs import credentials_field
from ._models import Design, Part
from ._testdata import TEST_DESIGN, TEST_PART, TEST_STEP_DATA_URI
from ._types import DesignStatus

CATEGORIES = {BlockCategory.HARDWARE, BlockCategory.DATA}


class RMFGDesignOutput(BlockSchemaOutput):
    design: Design = SchemaField(description="The design resource")
    design_id: str = SchemaField(description="ID to pass to DFM, quote and cart")
    status: DesignStatus = SchemaField(
        description="queued, processing, ready or failed"
    )
    parts: list[Part] = SchemaField(
        description="Every unique part with its instance count, once ready"
    )
    part: Part = SchemaField(description="One part at a time")
    part_ids: list[str] = SchemaField(description="Part IDs in the same order")
    review_url: str = SchemaField(
        description="Website page where a person can inspect and configure the design"
    )
    image_url: str = SchemaField(description="Rendered picture of the whole design")
    error: str = SchemaField(description="Error message if the request failed")


async def emit_design(design: Design) -> BlockOutput:
    yield "design", design
    yield "design_id", design.id
    yield "status", design.status
    yield "parts", design.parts
    for part in design.parts:
        yield "part", part
    yield "part_ids", [part.id for part in design.parts]
    if design.review_url:
        yield "review_url", design.review_url
    if design.image_url:
        yield "image_url", design.image_url


class RMFGAnalyzeDesignBlock(Block):
    """Upload a STEP file; RMFG splits it into parts and detects bends and holes."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        file: MediaFileType = SchemaField(
            description="STEP or STP file to analyze (URL, data URI or workspace file)"
        )
        file_name: str = SchemaField(
            description="Name to record for the upload; defaults to the file's own.",
            default="",
            advanced=True,
        )
        wait_for_ready: bool = SchemaField(
            description="Poll until analysis finishes instead of returning at once.",
            default=True,
            advanced=True,
        )
        timeout_seconds: int = SchemaField(
            description="How long to wait for analysis when wait_for_ready is on.",
            default=300,
            ge=5,
            le=1500,
            advanced=True,
        )
        idempotency_key: str = SchemaField(
            description=(
                "Stable key so a retried upload returns the same design. "
                "Defaults to this node execution's ID."
            ),
            default="",
            advanced=True,
        )

    class Output(RMFGDesignOutput):
        pass

    def __init__(self):
        super().__init__(
            id="bcbee3ee-4ff4-485b-afa8-a57856ab12b6",
            description="Uploads a STEP file to RMFG and returns its analyzed parts",
            categories=CATEGORIES,
            input_schema=RMFGAnalyzeDesignBlock.Input,
            output_schema=RMFGAnalyzeDesignBlock.Output,
            test_input={
                "file": TEST_STEP_DATA_URI,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("design", TEST_DESIGN),
                ("design_id", TEST_DESIGN.id),
                ("status", DesignStatus.READY),
                ("parts", [TEST_PART]),
                ("part", TEST_PART),
                ("part_ids", [TEST_PART.id]),
                ("review_url", TEST_DESIGN.review_url),
                ("image_url", TEST_DESIGN.image_url),
            ],
            test_mock={
                "read_step_file": lambda *args, **kwargs: ("bracket.step", b"ISO"),
                "analyze": lambda *args, **kwargs: TEST_DESIGN,
            },
        )

    @staticmethod
    async def read_step_file(
        file: MediaFileType, execution_context: ExecutionContext
    ) -> tuple[str, bytes]:
        """Materialise the input file locally and return (name, bytes)."""
        local_path = await store_media_file(
            file=file,
            execution_context=execution_context,
            return_format="for_local_processing",
        )
        if not execution_context.graph_exec_id:
            raise ValueError("execution_context.graph_exec_id is required")
        path = Path(get_exec_file_path(execution_context.graph_exec_id, local_path))
        return path.name, path.read_bytes()

    @staticmethod
    async def analyze(
        credentials: APIKeyCredentials,
        file_name: str,
        content: bytes,
        idempotency_key: str,
        wait_for_ready: bool,
        timeout_seconds: int,
    ) -> Design:
        client = RMFGClient(credentials)
        design = await client.analyze(file_name, content, idempotency_key)
        if not wait_for_ready:
            return design
        return await client.wait_for_design(design, timeout_seconds)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        node_exec_id: str = "",
        **kwargs,
    ) -> BlockOutput:
        file_name, content = await self.read_step_file(
            input_data.file, execution_context
        )
        if input_data.file_name:
            file_name = input_data.file_name
        if not file_name.lower().endswith((".step", ".stp")):
            file_name = f"{file_name}.step"
        design = await self.analyze(
            credentials,
            file_name,
            content,
            input_data.idempotency_key or node_exec_id,
            input_data.wait_for_ready,
            input_data.timeout_seconds,
        )
        async for output in emit_design(design):
            yield output


class RMFGGetDesignBlock(Block):
    """Read a design by ID, optionally waiting for analysis to finish."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        design_id: str = SchemaField(description="Design ID from Analyze Design")
        wait_for_ready: bool = SchemaField(
            description="Poll until analysis finishes instead of returning at once.",
            default=False,
            advanced=True,
        )
        timeout_seconds: int = SchemaField(
            description="How long to wait for analysis when wait_for_ready is on.",
            default=300,
            ge=5,
            le=1500,
            advanced=True,
        )

    class Output(RMFGDesignOutput):
        pass

    def __init__(self):
        super().__init__(
            id="8918fea0-ca1b-4df0-bcec-dfc66c66536e",
            description="Fetches an RMFG design and its analyzed parts by ID",
            categories=CATEGORIES,
            input_schema=RMFGGetDesignBlock.Input,
            output_schema=RMFGGetDesignBlock.Output,
            test_input={
                "design_id": TEST_DESIGN.id,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("design", TEST_DESIGN),
                ("design_id", TEST_DESIGN.id),
                ("status", DesignStatus.READY),
                ("parts", [TEST_PART]),
                ("part", TEST_PART),
                ("part_ids", [TEST_PART.id]),
                ("review_url", TEST_DESIGN.review_url),
                ("image_url", TEST_DESIGN.image_url),
            ],
            test_mock={"get_design": lambda *args, **kwargs: TEST_DESIGN},
        )

    @staticmethod
    async def get_design(
        credentials: APIKeyCredentials,
        design_id: str,
        wait_for_ready: bool,
        timeout_seconds: int,
    ) -> Design:
        client = RMFGClient(credentials)
        design = await client.get_design(design_id)
        if not wait_for_ready:
            return design
        return await client.wait_for_design(design, timeout_seconds)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        design = await self.get_design(
            credentials,
            input_data.design_id,
            input_data.wait_for_ready,
            input_data.timeout_seconds,
        )
        async for output in emit_design(design):
            yield output
