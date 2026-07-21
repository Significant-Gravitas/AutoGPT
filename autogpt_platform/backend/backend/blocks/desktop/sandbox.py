import time
from typing import Literal, Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.desktop._api import (
    WORKSPACE_PATH,
    DesktopSession,
    DesktopStream,
    PersistenceInfo,
)
from backend.blocks.desktop._common import (
    CREDENTIALS_FIELD_DESCRIPTION,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    WorkspaceScope,
    volume_name_for_scope,
)
from backend.blocks.desktop._cost import CostMeter, build_cost_meter
from backend.data.execution import ExecutionContext
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName


class CreateDesktopSandboxBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(description=CREDENTIALS_FIELD_DESCRIPTION)
        workspace_scope: WorkspaceScope = SchemaField(
            description=(
                "Which persistent workspace volume to mount: 'user' gives every "
                "sandbox you create the same durable workspace; 'agent' gives "
                "this agent its own durable workspace."
            ),
            default=WorkspaceScope.USER,
        )
        sandbox_id: str = SchemaField(
            description=(
                "Reconnect to an existing desktop sandbox instead of creating a "
                "new one. Suspended sandboxes resume automatically."
            ),
            default="",
            advanced=True,
        )
        width: int = SchemaField(description="Screen width in pixels", default=1280)
        height: int = SchemaField(description="Screen height in pixels", default=720)
        timeout_minutes: int = SchemaField(
            description=(
                "Idle timeout in minutes after which the desktop auto-suspends "
                "(state is preserved and resumes on reconnect)."
            ),
            default=15,
        )

    class Output(BlockSchemaOutput):
        sandbox_id: str = SchemaField(description="ID of the desktop sandbox")
        desktop_stream: DesktopStream = SchemaField(
            description=(
                "Live interactive desktop stream; the URL is directly embeddable "
                "and viewable in the browser."
            )
        )
        workspace_path: str = SchemaField(
            description="Path of the persistent workspace directory inside the sandbox"
        )
        persistence: PersistenceInfo = SchemaField(
            description="Whether a persistent volume is mounted for the workspace"
        )
        cost_meter: CostMeter = SchemaField(
            description="Estimated provider cost telemetry for this block run"
        )

    def __init__(self):
        super().__init__(
            id="a1e2b001-0001-4000-8000-de5c704b0001",
            description=(
                "Creates (or reconnects to) an interactive cloud desktop with a "
                "live, embeddable stream and a persistent mounted workspace."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CreateDesktopSandboxBlock.Input,
            output_schema=CreateDesktopSandboxBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "workspace_scope": WorkspaceScope.USER.value,
                "sandbox_id": "",
                "width": 1280,
                "height": 720,
                "timeout_minutes": 15,
            },
            test_output=[
                ("sandbox_id", "test-sandbox-id"),
                ("desktop_stream", lambda v: v["kind"] == "desktop_stream"),
                ("workspace_path", WORKSPACE_PATH),
                ("persistence", lambda v: v["volume_mounted"] is True),
                ("cost_meter", lambda v: v["provider"] == "e2b"),
            ],
            test_mock={
                "setup_desktop": lambda *args, **kwargs: (
                    "test-sandbox-id",
                    DesktopStream(
                        url="https://example.e2b.app/vnc.html",
                        sandbox_id="test-sandbox-id",
                    ),
                    PersistenceInfo(volume_mounted=True, volume_name="autogpt-user-x"),
                )
            },
        )

    async def setup_desktop(
        self,
        api_key: str,
        sandbox_id: str,
        width: int,
        height: int,
        timeout_minutes: int,
        volume_name: Optional[str],
    ) -> tuple[str, DesktopStream, PersistenceInfo]:
        if sandbox_id:
            session = await DesktopSession.connect(sandbox_id, api_key)
            mounted = await session.is_workspace_mounted()
            persistence = PersistenceInfo(
                volume_mounted=mounted, volume_name=volume_name if mounted else None
            )
        else:
            session, persistence = await DesktopSession.create(
                api_key=api_key,
                timeout_seconds=timeout_minutes * 60,
                width=width,
                height=height,
                volume_name=volume_name,
            )
        stream = await session.start_stream()
        return session.sandbox_id, stream, persistence

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        start = time.monotonic()
        try:
            sandbox_id, stream, persistence = await self.setup_desktop(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
                width=input_data.width,
                height=input_data.height,
                timeout_minutes=input_data.timeout_minutes,
                volume_name=volume_name_for_scope(
                    input_data.workspace_scope, execution_context
                ),
            )
            yield "sandbox_id", sandbox_id
            yield "desktop_stream", stream.model_dump()
            yield "workspace_path", WORKSPACE_PATH
            yield "persistence", persistence.model_dump()
            yield "cost_meter", build_cost_meter(
                sandbox_id, time.monotonic() - start
            ).model_dump()
        except Exception as e:
            yield "error", str(e)


class StopDesktopSandboxBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(description=CREDENTIALS_FIELD_DESCRIPTION)
        sandbox_id: str = SchemaField(description="ID of the desktop sandbox to stop")
        mode: Literal["suspend", "destroy"] = SchemaField(
            description=(
                "'suspend' preserves the full desktop state for later resume; "
                "'destroy' permanently deletes the sandbox (mounted workspace "
                "volumes survive)."
            ),
            default="suspend",
        )

    class Output(BlockSchemaOutput):
        sandbox_id: str = SchemaField(description="ID of the stopped sandbox")
        final_status: str = SchemaField(
            description="Resulting sandbox state: 'suspended' or 'destroyed'"
        )
        cost_meter: CostMeter = SchemaField(
            description="Estimated provider cost telemetry for this block run"
        )

    def __init__(self):
        super().__init__(
            id="a1e2b001-0005-4000-8000-de5c704b0005",
            description=(
                "Suspends (default) or destroys an interactive desktop sandbox. "
                "Suspended desktops preserve all state and can be resumed."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=StopDesktopSandboxBlock.Input,
            output_schema=StopDesktopSandboxBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "test-sandbox-id",
                "mode": "suspend",
            },
            test_output=[
                ("sandbox_id", "test-sandbox-id"),
                ("final_status", "suspended"),
                ("cost_meter", lambda v: v["provider"] == "e2b"),
            ],
            test_mock={"stop_desktop": lambda *args, **kwargs: "suspended"},
        )

    async def stop_desktop(self, api_key: str, sandbox_id: str, mode: str) -> str:
        session = await DesktopSession.connect(sandbox_id, api_key)
        if mode == "destroy":
            await session.kill()
            return "destroyed"
        await session.pause()
        return "suspended"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        start = time.monotonic()
        try:
            final_status = await self.stop_desktop(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
                mode=input_data.mode,
            )
            yield "sandbox_id", input_data.sandbox_id
            yield "final_status", final_status
            yield "cost_meter", build_cost_meter(
                input_data.sandbox_id, time.monotonic() - start
            ).model_dump()
        except Exception as e:
            yield "error", str(e)
