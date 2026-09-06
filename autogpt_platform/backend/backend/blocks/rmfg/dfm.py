"""Blocks that evaluate manufacturability (DFM) for a configured design."""

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
from ._models import DFMIssue, DFMReport, PartDFM, Requirement
from ._testdata import TEST_CONFIGURATION, TEST_DFM_ISSUE, TEST_DFM_REPORT
from ._types import ManufacturabilityStatus, ManufacturingConfiguration

CATEGORIES = {BlockCategory.HARDWARE, BlockCategory.DATA}


class RMFGDFMOutput(BlockSchemaOutput):
    report: DFMReport = SchemaField(description="The full DFM report")
    dfm_id: str = SchemaField(description="Report ID, for review links and re-reads")
    status: ManufacturabilityStatus = SchemaField(
        description="ready, requires_input (a selection is missing) or blocked"
    )
    is_ready: bool = SchemaField(description="True when nothing prevents ordering")
    configuration: ManufacturingConfiguration = SchemaField(
        description="The configuration that was evaluated; feed it to a quote"
    )
    issues: list[DFMIssue] = SchemaField(
        description="Every finding across all parts and the assembly"
    )
    issue: DFMIssue = SchemaField(description="One finding at a time")
    requirements: list[Requirement] = SchemaField(
        description="Selections still needed before the design can be quoted"
    )
    parts: list[PartDFM] = SchemaField(
        description="Per-part status, findings, capabilities and images"
    )
    review_url: str = SchemaField(
        description="Website page showing this exact configuration for a person to adjust"
    )
    error: str = SchemaField(description="Error message if the request failed")


async def emit_report(report: DFMReport) -> BlockOutput:
    yield "report", report
    yield "dfm_id", report.id
    yield "status", report.status
    yield "is_ready", report.status == ManufacturabilityStatus.READY
    yield "configuration", report.configuration
    issues = report.issues
    yield "issues", issues
    for issue in issues:
        yield "issue", issue
    yield "requirements", report.requirements
    yield "parts", report.parts
    if report.review_url:
        yield "review_url", report.review_url


class RMFGCreateDFMReportBlock(Block):
    """Check whether a configuration can be made, and learn what each part allows."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        design_id: str = SchemaField(description="Design ID from Analyze Design")
        material_id: str = SchemaField(
            description=(
                "Sheet-metal stock for every sheet part, from List Materials. "
                "Leave empty for tube-only designs or when configuration sets it."
            ),
            default="",
        )
        configuration: ManufacturingConfiguration = SchemaField(
            description=(
                "Per-part material, tube profile, finish, powder coat, hole "
                "operations, welds and accepted risks. A non-empty material_id "
                "above overrides defaults.material_id."
            ),
            default_factory=ManufacturingConfiguration,
            advanced=True,
        )
        generate_production_files: bool = SchemaField(
            description="Also prepare laser DXF and corrected STEP files.",
            default=True,
            advanced=True,
        )
        idempotency_key: str = SchemaField(
            description="Stable key for identical retries; defaults to the node execution ID.",
            default="",
            advanced=True,
        )

    class Output(RMFGDFMOutput):
        pass

    def __init__(self):
        super().__init__(
            id="0d755311-d01c-4cd9-a1cc-196a78590a18",
            description="Runs an RMFG manufacturability (DFM) check on a configured design",
            categories=CATEGORIES,
            input_schema=RMFGCreateDFMReportBlock.Input,
            output_schema=RMFGCreateDFMReportBlock.Output,
            test_input={
                "design_id": TEST_DFM_REPORT.design_id,
                "material_id": "mat_5052_0125",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("report", TEST_DFM_REPORT),
                ("dfm_id", TEST_DFM_REPORT.id),
                ("status", ManufacturabilityStatus.READY),
                ("is_ready", True),
                ("configuration", TEST_CONFIGURATION),
                ("issues", [TEST_DFM_ISSUE]),
                ("issue", TEST_DFM_ISSUE),
                ("requirements", []),
                ("parts", TEST_DFM_REPORT.parts),
                ("review_url", TEST_DFM_REPORT.review_url),
            ],
            test_mock={"create_report": lambda *args, **kwargs: TEST_DFM_REPORT},
        )

    @staticmethod
    async def create_report(
        credentials: APIKeyCredentials,
        design_id: str,
        configuration: ManufacturingConfiguration,
        generate_production_files: bool,
        idempotency_key: str,
    ) -> DFMReport:
        return await RMFGClient(credentials).create_dfm_report(
            design_id, configuration, generate_production_files, idempotency_key
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        node_exec_id: str = "",
        **kwargs,
    ) -> BlockOutput:
        report = await self.create_report(
            credentials,
            input_data.design_id,
            input_data.configuration.with_material(input_data.material_id),
            input_data.generate_production_files,
            input_data.idempotency_key or node_exec_id,
        )
        async for output in emit_report(report):
            yield output


class RMFGGetDFMReportBlock(Block):
    """Re-read a DFM report, e.g. to check whether production files are ready."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        dfm_id: str = SchemaField(description="Report ID from Create DFM Report")

    class Output(RMFGDFMOutput):
        pass

    def __init__(self):
        super().__init__(
            id="c36c49da-605f-4f1a-9da0-6f0a2cf9abfd",
            description="Fetches an RMFG DFM report by ID",
            categories=CATEGORIES,
            input_schema=RMFGGetDFMReportBlock.Input,
            output_schema=RMFGGetDFMReportBlock.Output,
            test_input={
                "dfm_id": TEST_DFM_REPORT.id,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("report", TEST_DFM_REPORT),
                ("dfm_id", TEST_DFM_REPORT.id),
                ("status", ManufacturabilityStatus.READY),
                ("is_ready", True),
                ("configuration", TEST_CONFIGURATION),
                ("issues", [TEST_DFM_ISSUE]),
                ("issue", TEST_DFM_ISSUE),
                ("requirements", []),
                ("parts", TEST_DFM_REPORT.parts),
                ("review_url", TEST_DFM_REPORT.review_url),
            ],
            test_mock={"get_report": lambda *args, **kwargs: TEST_DFM_REPORT},
        )

    @staticmethod
    async def get_report(credentials: APIKeyCredentials, dfm_id: str) -> DFMReport:
        return await RMFGClient(credentials).get_dfm_report(dfm_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        report = await self.get_report(credentials, input_data.dfm_id)
        async for output in emit_report(report):
            yield output
