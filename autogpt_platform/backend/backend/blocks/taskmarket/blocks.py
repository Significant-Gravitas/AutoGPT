from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from prisma.enums import ReviewStatus

from backend.blocks.helpers.review import HITLReviewHelper
from backend.blocks.taskmarket.cli import TaskMarketCLI
from backend.blocks.taskmarket.models import (
    TaskMarketCreationResult,
    TaskMarketMode,
    TaskMarketTaskPreview,
)
from backend.data.execution import ExecutionContext, ExecutionStatus
from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

class PrepareTaskMarketTaskBlock(Block):
    class Input(BlockSchemaInput):
        description: str = SchemaField(
            description="Exact public task description", min_length=1
        )
        deliverables: list[str] = SchemaField(
            description="Exact files or outcomes the worker must deliver"
        )
        reward_usdc: Decimal = SchemaField(
            description="USDC reward to escrow", gt=0
        )
        maximum_spend_usdc: Decimal = SchemaField(
            description="Hard operator-approved USDC spend ceiling", gt=0
        )
        deadline: datetime = SchemaField(
            description="Timezone-aware task deadline"
        )
        mode: TaskMarketMode = SchemaField(
            description="Task selection mode", default=TaskMarketMode.BOUNTY
        )
        tags: list[str] = SchemaField(
            description="Optional discovery tags", default_factory=list
        )

    class Output(BlockSchemaOutput):
        preview: TaskMarketTaskPreview = SchemaField(
            description="Immutable Base task preview for human authorization"
        )
        fingerprint: str = SchemaField(
            description="SHA-256 binding for every preview and spend field"
        )

    def __init__(self) -> None:
        super().__init__(
            id="11f5e907-e830-46f7-85c9-cad99fcafb15",
            description=(
                "Builds an immutable TaskMarket requester preview without moving funds"
            ),
            categories={BlockCategory.AGENT, BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "description": "Write an accessibility report",
                "deliverables": ["report.md"],
                "reward_usdc": "2",
                "maximum_spend_usdc": "2",
                "deadline": "2099-01-01T00:00:00Z",
                "mode": "bounty",
                "tags": ["accessibility"],
            },
            test_output=[
                ("preview", TaskMarketTaskPreview),
                ("fingerprint", lambda value: len(value) == 64),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        preview = TaskMarketTaskPreview.build(
            description=input_data.description,
            deliverables=input_data.deliverables,
            reward_usdc=input_data.reward_usdc,
            maximum_spend_usdc=input_data.maximum_spend_usdc,
            deadline=input_data.deadline,
            mode=input_data.mode.value,
            tags=input_data.tags,
        )
        yield "preview", preview
        yield "fingerprint", preview.fingerprint


class CreateTaskMarketTaskBlock(Block):
    class Input(BlockSchemaInput):
        preview: TaskMarketTaskPreview = SchemaField(
            description="Exact preview produced by PrepareTaskMarketTaskBlock"
        )

    class Output(BlockSchemaOutput):
        task_id: str = SchemaField(description="Created TaskMarket task ID")
        task_url: str = SchemaField(description="Canonical TaskMarket task link")
        live_status: dict[str, Any] = SchemaField(
            description="Live task state read back after creation"
        )

    def __init__(self) -> None:
        test_preview = TaskMarketTaskPreview.build(
            description="Write an accessibility report",
            deliverables=["report.md"],
            reward_usdc=Decimal("2"),
            maximum_spend_usdc=Decimal("2"),
            deadline=datetime(2099, 1, 1, tzinfo=timezone.utc),
            mode="bounty",
            tags=["accessibility"],
        )
        test_result = TaskMarketCreationResult(
            task_id="0x" + "1" * 64,
            task_url="https://taskmarket.dev/tasks/0x" + "1" * 64,
            live_status={"status": "open"},
        )
        super().__init__(
            id="84d76be7-c9f2-4481-9914-315c83347753",
            description=(
                "Creates one Base-funded TaskMarket task only after a fresh, exact "
                "human review"
            ),
            categories={BlockCategory.AGENT, BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"preview": test_preview.model_dump(mode="json")},
            test_output=[
                ("task_id", test_result.task_id),
                ("task_url", test_result.task_url),
                ("live_status", test_result.live_status),
            ],
            test_mock={
                "request_funding_approval": lambda **kwargs: True,
                "create_task": lambda preview: test_result,
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        node_id: str,
        node_exec_id: str,
        graph_exec_id: str,
        graph_id: str,
        graph_version: int,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        preview = input_data.preview
        self._validate_preview(preview)
        approved = await self.request_funding_approval(
            preview=preview,
            user_id=user_id,
            node_id=node_id,
            node_exec_id=node_exec_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            graph_version=graph_version,
            execution_context=execution_context,
        )
        if approved is None:
            return
        if not approved:
            raise self._execution_error("Task funding was rejected by the reviewer")
        result = await self.create_task(preview)
        yield "task_id", result.task_id
        yield "task_url", result.task_url
        yield "live_status", result.live_status

    async def request_funding_approval(
        self,
        *,
        preview: TaskMarketTaskPreview,
        user_id: str,
        node_id: str,
        node_exec_id: str,
        graph_exec_id: str,
        graph_id: str,
        graph_version: int,
        execution_context: ExecutionContext,
    ) -> bool | None:
        self._validate_execution_ids(
            user_id, node_id, node_exec_id, graph_exec_id, graph_id
        )
        payload = preview.model_dump(mode="json")
        result = await HITLReviewHelper.get_or_create_human_review(
            user_id=user_id,
            node_exec_id=node_exec_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            graph_version=graph_version,
            input_data=payload,
            message="Authorize this exact TaskMarket Base USDC funding request",
            editable=False,
            organization_id=execution_context.organization_id,
            team_id=execution_context.team_id,
        )
        if result is None:
            await HITLReviewHelper.update_node_execution_status(
                exec_id=node_exec_id, status=ExecutionStatus.REVIEW
            )
            return None
        if result.node_exec_id != node_exec_id or result.data != payload:
            raise self._execution_error("Review does not match this exact execution")
        await HITLReviewHelper.update_review_processed_status(node_exec_id, True)
        return result.status == ReviewStatus.APPROVED

    @staticmethod
    async def create_task(
        preview: TaskMarketTaskPreview,
    ) -> TaskMarketCreationResult:
        duration = preview.remaining_duration_hours(datetime.now(timezone.utc))
        return await TaskMarketCLI().create_and_read(
            description=preview.full_description(),
            reward_usdc=preview.reward_usdc,
            maximum_spend_usdc=preview.maximum_spend_usdc,
            duration_hours=duration,
            mode=preview.mode.value,
            tags=preview.tags,
        )

    def _validate_preview(self, preview: TaskMarketTaskPreview) -> None:
        try:
            preview.verify_fingerprint()
            preview.remaining_duration_hours(datetime.now(timezone.utc))
        except ValueError as error:
            raise BlockInputError(str(error), self.name, self.id) from error

    def _validate_execution_ids(self, *values: str) -> None:
        if any(not value for value in values):
            raise BlockInputError(
                "Task funding requires a persisted graph execution for fresh review",
                self.name,
                self.id,
            )

    def _execution_error(self, message: str) -> BlockExecutionError:
        return BlockExecutionError(message, self.name, self.id)


class InspectTaskMarketTaskBlock(Block):
    class Input(BlockSchemaInput):
        task_id: str = SchemaField(description="0x-prefixed TaskMarket task ID")

    class Output(BlockSchemaOutput):
        task: dict[str, Any] = SchemaField(description="Current live task state")
        submissions: list[dict[str, Any]] = SchemaField(
            description="Submissions presented for human review"
        )
        human_review_required: bool = SchemaField(
            description="Always true; this block cannot accept or reject work"
        )

    def __init__(self) -> None:
        task_id = "0x" + "1" * 64
        super().__init__(
            id="1f003c4e-21c7-4776-a467-2f52b23cc0a5",
            description=(
                "Reads a TaskMarket task and its submissions without deciding outcomes"
            ),
            categories={BlockCategory.AGENT, BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"task_id": task_id},
            test_output=[
                ("task", {"taskId": task_id, "status": "open"}),
                ("submissions", []),
                ("human_review_required", True),
            ],
            test_mock={
                "inspect_task": lambda task_id: (
                    {"taskId": task_id, "status": "open"},
                    [],
                )
            },
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        task, submissions = await self.inspect_task(input_data.task_id)
        yield "task", task
        yield "submissions", submissions
        yield "human_review_required", True

    @staticmethod
    async def inspect_task(
        task_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        cli = TaskMarketCLI()
        task = await cli.get_task(task_id)
        submissions = await cli.get_submissions(task_id)
        return task, submissions
