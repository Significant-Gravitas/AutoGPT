import time
from typing import Literal

from e2b import CommandExitException

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.desktop._api import WORKSPACE_PATH, DesktopSession
from backend.blocks.desktop._common import (
    CREDENTIALS_FIELD_DESCRIPTION,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
)
from backend.blocks.desktop._cost import CostMeter, build_cost_meter
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName


class DesktopCommandBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(description=CREDENTIALS_FIELD_DESCRIPTION)
        sandbox_id: str = SchemaField(
            description="ID of the desktop sandbox to run the command in"
        )
        command: str = SchemaField(
            description="Shell command to execute inside the desktop sandbox",
            placeholder="xdg-open https://example.com",
        )
        cwd: str = SchemaField(
            description="Working directory for the command",
            default=WORKSPACE_PATH,
            advanced=True,
        )
        timeout_seconds: int = SchemaField(
            description="Command timeout in seconds", default=60
        )

    class Output(BlockSchemaOutput):
        stdout: str = SchemaField(description="Standard output of the command")
        stderr: str = SchemaField(description="Standard error of the command")
        exit_code: int = SchemaField(description="Exit code of the command")
        cost_meter: CostMeter = SchemaField(
            description="Estimated provider cost telemetry for this block run"
        )

    def __init__(self):
        super().__init__(
            id="a1e2b001-0003-4000-8000-de5c704b0003",
            description=(
                "Runs a shell command inside an interactive desktop sandbox "
                "(DISPLAY is set, so GUI apps can be launched)."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=DesktopCommandBlock.Input,
            output_schema=DesktopCommandBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "test-sandbox-id",
                "command": "echo hello",
                "timeout_seconds": 60,
            },
            test_output=[
                ("stdout", "hello\n"),
                ("stderr", ""),
                ("exit_code", 0),
                ("cost_meter", lambda v: v["provider"] == "e2b"),
            ],
            test_mock={"exec_command": lambda *args, **kwargs: ("hello\n", "", 0)},
        )

    async def exec_command(
        self, api_key: str, sandbox_id: str, command: str, cwd: str, timeout: int
    ) -> tuple[str, str, int]:
        session = await DesktopSession.connect(sandbox_id, api_key)
        try:
            result = await session.run_command(command, cwd=cwd, timeout=timeout)
        except CommandExitException as exc:
            return exc.stdout, exc.stderr, exc.exit_code
        return result.stdout, result.stderr, result.exit_code

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        start = time.monotonic()
        try:
            stdout, stderr, exit_code = await self.exec_command(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
                command=input_data.command,
                cwd=input_data.cwd,
                timeout=input_data.timeout_seconds,
            )
            yield "stdout", stdout
            yield "stderr", stderr
            yield "exit_code", exit_code
            yield "cost_meter", build_cost_meter(
                input_data.sandbox_id, time.monotonic() - start
            ).model_dump()
        except Exception as e:
            yield "error", str(e)
