import asyncio
import json
import re
import shutil
from collections.abc import Awaitable, Callable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from backend.blocks.taskmarket.models import (
    BASE_CHAIN_ID,
    BASE_USDC_ADDRESS,
    TaskMarketCreationResult,
    TaskMarketPreflight,
)

JsonResult = dict[str, Any] | list[Any]
CommandRunner = Callable[[Sequence[str]], Awaitable[JsonResult]]
TASK_ID_PATTERN = re.compile(r"^0x[0-9a-fA-F]{64}$")
WALLET_PATTERN = re.compile(r"^0x[0-9a-fA-F]{40}$")


class TaskMarketCLIError(RuntimeError):
    """The first-party TaskMarket CLI could not complete a safe operation."""


class SettlementUnknownError(TaskMarketCLIError):
    """A funding call ended without a trustworthy settlement result."""


class TaskMarketCLI:
    def __init__(
        self,
        command_runner: CommandRunner | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self._runner = command_runner or self._run_command
        self._timeout_seconds = timeout_seconds

    async def preflight(self, maximum_spend_usdc: Decimal) -> TaskMarketPreflight:
        deposit = self._expect_object(await self._runner(("deposit",)))
        legal = self._expect_object(await self._runner(("legal", "status")))
        balance = self._expect_object(await self._runner(("wallet", "balance")))
        preflight = TaskMarketPreflight(
            wallet_address=str(deposit.get("address", "")),
            chain_id=int(deposit.get("chainId", 0)),
            usdc_contract=str(deposit.get("usdcContract", "")),
            balance_usdc=Decimal(str(balance.get("balanceUsdc", "0"))),
            legal_accepted=legal.get("accepted") is True,
        )
        self._validate_preflight(preflight, maximum_spend_usdc)
        return preflight

    async def create_task(
        self,
        *,
        description: str,
        reward_usdc: Decimal,
        duration_hours: Decimal,
        mode: str,
        tags: list[str],
    ) -> str:
        args = (
            "task",
            "create",
            "--description",
            description,
            "--reward",
            format(reward_usdc, "f"),
            "--duration",
            format(duration_hours, "f"),
            "--mode",
            mode,
            "--tags",
            ",".join(tags),
        )
        try:
            result = self._expect_object(await self._runner(args))
        except Exception as error:
            raise SettlementUnknownError(
                "Task creation settlement is unknown and must not be retried"
            ) from error
        task_id = str(result.get("taskId", ""))
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise SettlementUnknownError(
                "Task creation returned no verifiable task ID; settlement is unknown "
                "and must not be retried"
            )
        return task_id

    async def create_and_read(
        self,
        *,
        description: str,
        reward_usdc: Decimal,
        maximum_spend_usdc: Decimal,
        duration_hours: Decimal,
        mode: str,
        tags: list[str],
    ) -> TaskMarketCreationResult:
        await self.preflight(maximum_spend_usdc)
        task_id = await self.create_task(
            description=description,
            reward_usdc=reward_usdc,
            duration_hours=duration_hours,
            mode=mode,
            tags=tags,
        )
        try:
            status = await self.get_task(task_id)
        except Exception:
            status = {
                "taskId": task_id,
                "status": "unknown",
                "humanAction": "Open the task link before attempting any new write",
            }
        return TaskMarketCreationResult(
            task_id=task_id,
            task_url=f"https://taskmarket.dev/tasks/{task_id}",
            live_status=status,
        )

    async def get_task(self, task_id: str) -> dict[str, Any]:
        self._validate_task_id(task_id)
        result = await self._runner(("task", "get", task_id))
        return self._expect_object(result)

    async def get_submissions(self, task_id: str) -> list[dict[str, Any]]:
        self._validate_task_id(task_id)
        result = await self._runner(("task", "submissions", task_id))
        if isinstance(result, list):
            return [self._expect_object(item) for item in result]
        submissions = self._expect_object(result).get("submissions", [])
        if not isinstance(submissions, list):
            raise TaskMarketCLIError("CLI returned invalid submission data")
        return [self._expect_object(item) for item in submissions]

    async def _run_command(self, arguments: Sequence[str]) -> JsonResult:
        executable = self._resolve_executable()
        process = await asyncio.create_subprocess_exec(
            executable,
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        if process.returncode != 0:
            raise TaskMarketCLIError("TaskMarket CLI command failed")
        return self._parse_output(stdout)

    @staticmethod
    def _resolve_executable() -> str:
        executable = shutil.which("taskmarket")
        if not executable:
            raise TaskMarketCLIError(
                "First-party TaskMarket CLI was not found on the server PATH"
            )
        if Path(executable).suffix.lower() in {".bat", ".cmd"}:
            raise TaskMarketCLIError(
                "TaskMarket CLI must be a direct executable, not a shell wrapper"
            )
        return executable

    @staticmethod
    def _parse_output(stdout: bytes) -> JsonResult:
        if len(stdout) > 1_000_000:
            raise TaskMarketCLIError("TaskMarket CLI output exceeded the safe limit")
        try:
            envelope = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TaskMarketCLIError("TaskMarket CLI returned invalid JSON") from error
        if not isinstance(envelope, dict) or envelope.get("ok") is not True:
            raise TaskMarketCLIError("TaskMarket CLI returned an unsuccessful response")
        data = envelope.get("data")
        if not isinstance(data, (dict, list)):
            raise TaskMarketCLIError("TaskMarket CLI returned invalid result data")
        return data

    @staticmethod
    def _expect_object(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise TaskMarketCLIError("TaskMarket CLI returned an invalid object")
        return result

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("Task ID must be a 0x-prefixed 32-byte hex value")

    @staticmethod
    def _validate_preflight(
        preflight: TaskMarketPreflight, maximum_spend_usdc: Decimal
    ) -> None:
        if not WALLET_PATTERN.fullmatch(preflight.wallet_address):
            raise TaskMarketCLIError("TaskMarket CLI returned an invalid wallet address")
        if preflight.chain_id != BASE_CHAIN_ID:
            raise TaskMarketCLIError("Task creation is restricted to Base chain 8453")
        if preflight.usdc_contract.lower() != BASE_USDC_ADDRESS.lower():
            raise TaskMarketCLIError("Task creation requires canonical Base USDC")
        if not preflight.legal_accepted:
            raise TaskMarketCLIError("Current TaskMarket terms require operator acceptance")
        if preflight.balance_usdc < maximum_spend_usdc:
            raise TaskMarketCLIError("Wallet balance is below the approved maximum spend")
