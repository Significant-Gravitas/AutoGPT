"""Blocks that start a Capy pull request review and read its findings."""

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

from ._api import CapyClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, capy_credentials_field
from ._testdata import TEST_REVIEW_ROUND, TEST_REVIEW_STARTED
from ._types import ReviewFinding, ReviewRound, ReviewStarted, ReviewTier


class CapyStartReviewBlock(Block):
    """Start a Capy review round on a GitHub pull request."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        repo: str = SchemaField(
            description="The repository as owner/name",
            placeholder="acme/checkout",
        )
        pr_number: int = SchemaField(description="The pull request number", ge=1)
        tier: ReviewTier = SchemaField(
            description=(
                "Review depth and cost: low, medium or high. Empty uses the "
                "repository's default."
            ),
            default=ReviewTier.DEFAULT,
        )
        idempotency_key: str = SchemaField(
            description=(
                "Retry key: the same key returns the same round instead of "
                "starting another. Also the request_id you read the round back "
                "with. Empty lets Capy key the round on the PR's exact commits."
            ),
            default="",
            advanced=True,
        )
        source_thread_id: str = SchemaField(
            description=(
                "A Capy thread that should receive the verdict and triage the "
                "findings itself"
            ),
            default="",
            advanced=True,
        )
        force_refresh: bool = SchemaField(
            description="Re-run a round that already completed on the same commits",
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        review: ReviewStarted = SchemaField(
            description="The started (or adopted) round"
        )
        request_id: str = SchemaField(
            description="Pass to Capy Get Review Round to read the findings"
        )
        adopted: bool = SchemaField(
            description=(
                "True when a round already existed for these exact commits and "
                "nothing new was started"
            )
        )

    def __init__(self):
        super().__init__(
            id="41c3185c-43fd-4b76-bf4f-6b8e97522a17",
            description=(
                "Starts a Capy code review on a GitHub pull request. Capy's "
                "review agent reads the diff in a real checkout and posts "
                "findings as inline comments on the pull request. Bills the "
                "review to your Capy organization."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyStartReviewBlock.Input,
            output_schema=CapyStartReviewBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "repo": "acme/checkout",
                "pr_number": 481,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("review", TEST_REVIEW_STARTED),
                ("request_id", TEST_REVIEW_STARTED.request_id),
                ("adopted", False),
            ],
            test_mock={"start_review": lambda *args, **kwargs: TEST_REVIEW_STARTED},
            # The findings post as comments on the pull request, visible to
            # everyone on it.
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def start_review(
        credentials: APIKeyCredentials, input_data: "CapyStartReviewBlock.Input"
    ) -> ReviewStarted:
        return await CapyClient(credentials).start_review(
            repo=input_data.repo,
            pr_number=input_data.pr_number,
            idempotency_key=input_data.idempotency_key,
            tier=input_data.tier.value,
            source_thread_id=input_data.source_thread_id,
            force_refresh=input_data.force_refresh,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        review = await self.start_review(credentials, input_data)
        yield "review", review
        yield "request_id", review.request_id
        yield "adopted", review.adopted


class CapyGetReviewRoundBlock(Block):
    """Read a Capy review round's status and findings."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        request_id: str = SchemaField(
            description="The request_id Capy Start Review returned"
        )

    class Output(BlockSchemaOutput):
        round: ReviewRound = SchemaField(description="The review round")
        status: str = SchemaField(
            description="pending, running, completed, failed or stale"
        )
        is_settled: bool = SchemaField(
            description="True once the round has completed, failed or gone stale"
        )
        findings: list[ReviewFinding] = SchemaField(
            description="Findings with severity, confidence, category, file and line"
        )
        high_severity_count: int = SchemaField(
            description="Number of high-severity issues"
        )

    def __init__(self):
        super().__init__(
            id="8cf41c84-e897-4552-bbde-3403c21b0ac8",
            description=(
                "Gets a Capy review round's status and its findings, each with "
                "severity, confidence, category, file and line."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyGetReviewRoundBlock.Input,
            output_schema=CapyGetReviewRoundBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "request_id": TEST_REVIEW_ROUND.request_id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("round", TEST_REVIEW_ROUND),
                ("status", "completed"),
                ("is_settled", True),
                ("findings", TEST_REVIEW_ROUND.findings),
                ("high_severity_count", 1),
            ],
            test_mock={"get_review_round": lambda *args, **kwargs: TEST_REVIEW_ROUND},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def get_review_round(
        credentials: APIKeyCredentials, request_id: str
    ) -> ReviewRound:
        return await CapyClient(credentials).get_review_round(request_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        review_round = await self.get_review_round(credentials, input_data.request_id)
        yield "round", review_round
        yield "status", review_round.status
        yield "is_settled", review_round.status in {"completed", "failed", "stale"}
        yield "findings", review_round.findings
        yield "high_severity_count", sum(
            1
            for f in review_round.findings
            if f.kind == "issue" and f.severity == "high"
        )
