import asyncio
import logging
import time
import uuid

from pydantic import BaseModel
from tenki import AsyncClient, AsyncSandbox, CommandResult

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

from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, _client, tenki

logger = logging.getLogger(__name__)


class SandboxExecution(BaseModel):
    sandbox_id: str
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    startup_time_seconds: float
    ok: bool
    failure: str = ""


class TenkiRunCodeBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = tenki.credentials_field(
            description="Tenki API key from https://app.tenki.cloud"
        )
        command: str = SchemaField(
            description="Shell command to run in a fresh Tenki sandbox",
            placeholder="python3 -c \"print('hello from Tenki')\"",
            min_length=1,
        )
        working_directory: str = SchemaField(
            description="Sandbox working directory; empty uses /home/tenki",
            default="",
            advanced=True,
        )
        environment: dict[str, str] = SchemaField(
            description="Environment variables passed only to the command",
            default_factory=dict,
            advanced=True,
        )
        timeout_seconds: int = SchemaField(
            description="Maximum command runtime in seconds",
            default=120,
            ge=1,
            le=900,
            advanced=True,
        )
        startup_timeout_seconds: int = SchemaField(
            description="Maximum time to wait for the sandbox to become ready",
            default=180,
            ge=30,
            le=300,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        stdout: str = SchemaField(description="Command standard output")
        stderr: str = SchemaField(description="Command standard error")
        exit_code: int = SchemaField(description="Command exit code")
        duration_seconds: float = SchemaField(
            description="Command execution time in seconds"
        )
        startup_time_seconds: float = SchemaField(
            description="Sandbox create-to-ready time in seconds"
        )
        sandbox_id: str = SchemaField(
            description="ID of the terminated ephemeral sandbox"
        )

    def __init__(self):
        super().__init__(
            id="327a0cac-3947-4d90-a32c-66509bf156c4",
            description=(
                "Run a shell command in a fresh Tenki cloud sandbox. The sandbox "
                "is always terminated after the command finishes or fails."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "command": "printf 'hello from Tenki'",
            },
            test_output=[
                ("stdout", "hello from Tenki"),
                ("stderr", ""),
                ("exit_code", 0),
                ("duration_seconds", 0.1),
                ("startup_time_seconds", 1.0),
                ("sandbox_id", "sandbox-id"),
            ],
            test_mock={
                "execute_in_sandbox": lambda *args, **kwargs: SandboxExecution(
                    sandbox_id="sandbox-id",
                    stdout="hello from Tenki",
                    stderr="",
                    exit_code=0,
                    duration_seconds=0.1,
                    startup_time_seconds=1.0,
                    ok=True,
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self.execute_in_sandbox(input_data, credentials)
        except Exception as error:
            yield "error", f"Tenki sandbox execution failed: {error}"
            return

        if not result.ok:
            yield "error", result.failure
            return

        yield "stdout", result.stdout
        yield "stderr", result.stderr
        yield "exit_code", result.exit_code
        yield "duration_seconds", result.duration_seconds
        yield "startup_time_seconds", result.startup_time_seconds
        yield "sandbox_id", result.sandbox_id

    async def execute_in_sandbox(
        self, input_data: Input, credentials: APIKeyCredentials
    ) -> SandboxExecution:
        client = _client(credentials)
        sandbox: AsyncSandbox | None = None
        try:
            started_at = time.monotonic()
            sandbox = await client.create(
                name=f"autogpt-{uuid.uuid4().hex[:12]}",
                wait=False,
                allow_inbound=False,
                max_duration=(
                    input_data.startup_timeout_seconds + input_data.timeout_seconds + 60
                ),
                metadata={"integration": "autogpt"},
                tags=["autogpt", "ephemeral"],
            )
            await sandbox.wait_ready(timeout=input_data.startup_timeout_seconds)
            startup_seconds = time.monotonic() - started_at
            result = await sandbox.shell(
                input_data.command,
                cwd=input_data.working_directory or None,
                env=input_data.environment,
                timeout=input_data.timeout_seconds,
            )
            return SandboxExecution(
                sandbox_id=sandbox.id,
                stdout=result.stdout_text,
                stderr=result.stderr_text,
                exit_code=result.exit_code,
                duration_seconds=(result.duration_ms or 0) / 1000,
                startup_time_seconds=startup_seconds,
                ok=result.ok,
                failure=self._command_failure(result),
            )
        finally:
            await self._cleanup(client, sandbox)

    @staticmethod
    async def _cleanup(client: AsyncClient, sandbox: AsyncSandbox | None) -> None:
        """Best-effort teardown; must not mask the run's result or exception."""

        async def close_resources() -> None:
            try:
                if sandbox is not None:
                    await sandbox.close_if_open()
            except asyncio.CancelledError:
                raise
            except Exception:
                sandbox_id = sandbox.id if sandbox else ""
                logger.warning(
                    f"Failed to close Tenki sandbox {sandbox_id}", exc_info=True
                )
            finally:
                try:
                    await client.close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("Failed to close Tenki client", exc_info=True)

        cleanup_task = asyncio.create_task(close_resources())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise

    @staticmethod
    def _command_failure(result: CommandResult) -> str:
        if result.ok:
            return ""
        details = (
            result.stderr_text.strip()
            or result.stdout_text.strip()
            or result.reason
            or result.signal
            or "no diagnostics returned"
        )
        return f"Tenki command exited with code {result.exit_code}: {details}"
