"""
E2B Desktop Sandbox blocks for AutoGPT.

These blocks expose the parts of E2B Desktop that a headless code sandbox
cannot do: a full graphical Linux desktop (Ubuntu + XFCE) with a live VNC
stream and mouse/keyboard control. This is the "computer use" surface —
an agent sees the screen and drives it like a human would.

For running code or shell commands in a headless sandbox, use the Code
Executor blocks (`code_executor.py`) instead — those cover sandbox
creation, command execution and teardown for the non-visual case.

Blocks:
  - E2BDesktopCreateBlock     : Create a desktop sandbox + start a live stream
  - E2BDesktopControlBlock    : Drive mouse + keyboard (click / type / press / scroll)
  - E2BDesktopScreenshotBlock : Capture the desktop screen as an image
  - E2BDesktopPauseBlock       : Pause the sandbox (keep state, stop compute billing)
  - E2BDesktopKillBlock        : Destroy the desktop sandbox and stop billing

The E2B Desktop SDK is synchronous, so every SDK call is dispatched to a
worker thread via ``asyncio.to_thread`` to avoid blocking the executor's
event loop.
"""

import asyncio
import base64
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional, Protocol, cast

from e2b_desktop import Sandbox
from pydantic import SecretStr

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.file import store_media_file
from backend.util.type import MediaFileType

if TYPE_CHECKING:
    from backend.executor.utils import ExecutionContext

# Shared test credentials (same pattern as code_executor.py)
TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcde0",
    provider="e2b",
    api_key=SecretStr("mock-e2b-api-key"),
    title="Mock E2B API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


# The e2b-desktop SDK ships no `py.typed` marker, so its desktop-only methods
# are invisible to the type checker. This Protocol describes just the surface
# we use, letting us `cast()` the SDK objects instead of suppressing errors.
class _StreamHandle(Protocol):
    def start(self, *, require_auth: bool = ...) -> None: ...
    def get_auth_key(self) -> str: ...
    def get_url(self, *, auth_key: Optional[str] = ...) -> str: ...


class _CommandsHandle(Protocol):
    def run(self, cmd: str) -> object: ...


class _DesktopSandbox(Protocol):
    sandbox_id: str
    stream: _StreamHandle
    commands: _CommandsHandle

    def left_click(self, x: Optional[int] = ..., y: Optional[int] = ...) -> None: ...
    def double_click(self, x: Optional[int] = ..., y: Optional[int] = ...) -> None: ...
    def right_click(self, x: Optional[int] = ..., y: Optional[int] = ...) -> None: ...
    def middle_click(self, x: Optional[int] = ..., y: Optional[int] = ...) -> None: ...
    def move_mouse(self, x: int, y: int) -> None: ...
    def scroll(self, direction: str = ..., amount: int = ...) -> None: ...
    def write(self, text: str) -> None: ...
    def press(self, key: str | list[str]) -> None: ...
    def screenshot(self) -> bytes: ...
    def pause(self) -> Optional[str]: ...
    def kill(self) -> None: ...


def _create_sandbox(
    api_key: str,
    template: Optional[str],
    timeout: int,
    resolution: tuple[int, int],
    dpi: int,
) -> _DesktopSandbox:
    return cast(
        _DesktopSandbox,
        Sandbox.create(
            api_key=api_key,
            template=template,
            timeout=timeout,
            resolution=resolution,
            dpi=dpi,
        ),
    )


def _connect_sandbox(api_key: str, sandbox_id: str) -> _DesktopSandbox:
    return cast(
        _DesktopSandbox, Sandbox.connect(sandbox_id=sandbox_id, api_key=api_key)
    )


class E2BDesktopCreateBlock(Block):
    """
    Spin up an E2B Desktop sandbox and start a live browser stream.

    The returned ``stream_url`` can be embedded as an iframe so users watch the
    desktop in real time. Use ``template_id`` to point at a pre-baked E2B
    template with your dependencies already installed to cut cold-start time.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description="E2B API key. Get yours at https://e2b.dev/docs",
        )
        template_id: str = SchemaField(
            description=(
                "E2B desktop template ID. Use a pre-baked template with your "
                "dependencies already installed to skip setup on every run. "
                "Leave empty for the default desktop template."
            ),
            default="",
            advanced=True,
        )
        setup_commands: list[str] = SchemaField(
            description=(
                "Shell commands to run after the desktop starts, e.g. "
                "['git clone https://github.com/you/app /home/user/app']. "
                "Prefer baking these into a template_id for faster cold starts."
            ),
            default_factory=list,
            advanced=False,
        )
        timeout: int = SchemaField(
            description=(
                "Sandbox lifetime in seconds. Max 3600 (1 hr) on Hobby, "
                "86400 (24 hr) on Pro. The sandbox auto-kills after this — use "
                "the Kill Desktop Sandbox block to stop early and save cost."
            ),
            default=3600,
        )
        stream_require_auth: bool = SchemaField(
            description=(
                "Require an auth key to view the live stream. Leave True in "
                "production — disabling exposes the desktop to anyone with the URL."
            ),
            default=True,
            advanced=True,
        )
        width: int = SchemaField(
            description=(
                "Desktop screen width in pixels. Lower resolutions stream more "
                "smoothly (fewer pixels per frame); higher ones are sharper."
            ),
            default=1280,
            advanced=True,
        )
        height: int = SchemaField(
            description="Desktop screen height in pixels.",
            default=720,
            advanced=True,
        )
        dpi: int = SchemaField(
            description=(
                "Desktop DPI (dots per inch). Raise it to scale up UI on "
                "high-resolution screens; 96 is the standard default."
            ),
            default=96,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        sandbox_id: str = SchemaField(
            description=(
                "ID of the running desktop sandbox. Pass this to the other "
                "E2B Desktop blocks to control or capture it."
            )
        )
        stream_url: str = SchemaField(
            description="Live browser-accessible stream URL. Embed as an iframe."
        )
        auth_key: str = SchemaField(
            description=(
                "Auth key required to view the stream (when "
                "stream_require_auth=True). Already included in stream_url."
            )
        )
        error: str = SchemaField(description="Error message if creation failed.")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description=(
                "Create an E2B Desktop sandbox (a full Linux GUI desktop) and "
                "start a live browser stream. Returns a stream_url you can embed "
                "to watch the desktop in real time."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=E2BDesktopCreateBlock.Input,
            output_schema=E2BDesktopCreateBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "template_id": "",
                "setup_commands": [],
                "timeout": 3600,
                "stream_require_auth": True,
                "width": 1280,
                "height": 720,
                "dpi": 96,
            },
            test_output=[
                ("sandbox_id", "mock-sandbox-id"),
                ("stream_url", "https://mock-stream.e2b.dev/stream?auth=mock-key"),
                ("auth_key", "mock-auth-key"),
            ],
            test_mock={
                "create_desktop": lambda *args, **kwargs: (
                    "mock-sandbox-id",
                    "https://mock-stream.e2b.dev/stream?auth=mock-key",
                    "mock-auth-key",
                )
            },
        )

    @staticmethod
    def _create_desktop(
        api_key: str,
        template_id: str,
        setup_commands: list[str],
        timeout: int,
        stream_require_auth: bool,
        resolution: tuple[int, int],
        dpi: int,
    ) -> tuple[str, str, str]:
        desktop = _create_sandbox(
            api_key, template_id or None, timeout, resolution, dpi
        )
        for cmd in setup_commands:
            desktop.commands.run(cmd)

        desktop.stream.start(require_auth=stream_require_auth)
        auth_key = desktop.stream.get_auth_key() if stream_require_auth else ""
        stream_url = desktop.stream.get_url(auth_key=auth_key or None)
        return desktop.sandbox_id, stream_url, auth_key

    async def create_desktop(
        self,
        api_key: str,
        template_id: str,
        setup_commands: list[str],
        timeout: int,
        stream_require_auth: bool,
        resolution: tuple[int, int],
        dpi: int,
    ) -> tuple[str, str, str]:
        return await asyncio.to_thread(
            self._create_desktop,
            api_key,
            template_id,
            setup_commands,
            timeout,
            stream_require_auth,
            resolution,
            dpi,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            sandbox_id, stream_url, auth_key = await self.create_desktop(
                api_key=credentials.api_key.get_secret_value(),
                template_id=input_data.template_id,
                setup_commands=input_data.setup_commands,
                timeout=input_data.timeout,
                stream_require_auth=input_data.stream_require_auth,
                resolution=(input_data.width, input_data.height),
                dpi=input_data.dpi,
            )
            yield "sandbox_id", sandbox_id
            yield "stream_url", stream_url
            yield "auth_key", auth_key
        except Exception as e:
            yield "error", str(e)


class DesktopAction(Enum):
    LEFT_CLICK = "left_click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    MOVE_MOUSE = "move_mouse"
    SCROLL = "scroll"
    TYPE = "type"
    PRESS = "press"


class ScrollDirection(Enum):
    UP = "up"
    DOWN = "down"


class E2BDesktopControlBlock(Block):
    """
    Drive the mouse and keyboard inside a running E2B Desktop sandbox.

    This is the "act" half of a computer-use loop: pair it with the Screenshot
    block to see the screen, decide an action, perform it, then screenshot again.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description="E2B API key — must match the key used to create the sandbox.",
        )
        sandbox_id: str = SchemaField(
            description="Sandbox ID from the Create Desktop Sandbox block.",
        )
        action: DesktopAction = SchemaField(
            description=(
                "Which input action to perform: click variants, move the mouse, "
                "scroll, type text, or press a key/combo."
            ),
            default=DesktopAction.LEFT_CLICK,
        )
        x: Optional[int] = SchemaField(
            description=(
                "X coordinate for click/move actions. Leave empty for clicks to "
                "use the current cursor position. Required for move_mouse."
            ),
            default=None,
        )
        y: Optional[int] = SchemaField(
            description=(
                "Y coordinate for click/move actions. Leave empty for clicks to "
                "use the current cursor position. Required for move_mouse."
            ),
            default=None,
        )
        text: str = SchemaField(
            description="Text to type (used by the 'type' action).",
            default="",
        )
        keys: str = SchemaField(
            description=(
                "Key or key combo to press (used by the 'press' action), e.g. "
                "'enter', 'backspace', or 'ctrl+c' for a combination."
            ),
            default="",
        )
        scroll_direction: ScrollDirection = SchemaField(
            description="Scroll direction (used by the 'scroll' action).",
            default=ScrollDirection.DOWN,
            advanced=True,
        )
        scroll_amount: int = SchemaField(
            description="Number of scroll steps (used by the 'scroll' action).",
            default=3,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="True if the action was performed.")
        error: str = SchemaField(description="Error message if the action failed.")

    def __init__(self):
        super().__init__(
            id="f6a7b8c9-d0e1-2345-fabc-456789012345",
            description=(
                "Control the mouse and keyboard of a running E2B Desktop sandbox: "
                "click, move, scroll, type text, or press keys. Pair with the "
                "Screenshot block to build a see-then-act computer-use loop."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=E2BDesktopControlBlock.Input,
            output_schema=E2BDesktopControlBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "mock-sandbox-id",
                "action": DesktopAction.LEFT_CLICK.value,
                "x": 100,
                "y": 200,
                "text": "",
                "keys": "",
                "scroll_direction": ScrollDirection.DOWN.value,
                "scroll_amount": 3,
            },
            test_output=[
                ("success", True),
            ],
            test_mock={"perform_action": lambda *args, **kwargs: None},
        )

    @staticmethod
    def _perform_action(
        api_key: str,
        sandbox_id: str,
        action: DesktopAction,
        x: Optional[int],
        y: Optional[int],
        text: str,
        keys: str,
        scroll_direction: ScrollDirection,
        scroll_amount: int,
    ) -> None:
        desktop = _connect_sandbox(api_key, sandbox_id)

        if action is DesktopAction.LEFT_CLICK:
            desktop.left_click(x=x, y=y)
        elif action is DesktopAction.DOUBLE_CLICK:
            desktop.double_click(x=x, y=y)
        elif action is DesktopAction.RIGHT_CLICK:
            desktop.right_click(x=x, y=y)
        elif action is DesktopAction.MIDDLE_CLICK:
            desktop.middle_click(x=x, y=y)
        elif action is DesktopAction.MOVE_MOUSE:
            if x is None or y is None:
                raise ValueError("move_mouse requires both x and y coordinates.")
            desktop.move_mouse(x, y)
        elif action is DesktopAction.SCROLL:
            desktop.scroll(direction=scroll_direction.value, amount=scroll_amount)
        elif action is DesktopAction.TYPE:
            desktop.write(text)
        elif action is DesktopAction.PRESS:
            if not keys:
                raise ValueError("press requires a key or key combo in 'keys'.")
            combo = [k.strip() for k in keys.split("+") if k.strip()]
            desktop.press(combo if len(combo) > 1 else combo[0])

    async def perform_action(
        self,
        api_key: str,
        sandbox_id: str,
        action: DesktopAction,
        x: Optional[int],
        y: Optional[int],
        text: str,
        keys: str,
        scroll_direction: ScrollDirection,
        scroll_amount: int,
    ) -> None:
        await asyncio.to_thread(
            self._perform_action,
            api_key,
            sandbox_id,
            action,
            x,
            y,
            text,
            keys,
            scroll_direction,
            scroll_amount,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            await self.perform_action(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
                action=input_data.action,
                x=input_data.x,
                y=input_data.y,
                text=input_data.text,
                keys=input_data.keys,
                scroll_direction=input_data.scroll_direction,
                scroll_amount=input_data.scroll_amount,
            )
            yield "success", True
        except Exception as e:
            yield "error", str(e)
            yield "success", False


class E2BDesktopScreenshotBlock(Block):
    """
    Capture a screenshot of the current E2B Desktop sandbox screen.

    The image is stored in the AutoGPT workspace via ``store_media_file`` so it
    can be passed to other blocks (e.g. a vision model) or embedded in outputs.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description="E2B API key — must match the key used to create the sandbox.",
        )
        sandbox_id: str = SchemaField(
            description="Sandbox ID from the Create Desktop Sandbox block.",
        )

    class Output(BlockSchemaOutput):
        image: MediaFileType = SchemaField(
            description=(
                "The captured screenshot. A workspace reference in CoPilot, or a "
                "data URI in graphs — feed it directly into downstream blocks."
            )
        )
        error: str = SchemaField(description="Error message if the screenshot failed.")

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-0123-defa-234567890123",
            description=(
                "Capture a PNG screenshot of the current E2B Desktop sandbox screen "
                "and store it in the AutoGPT workspace. Use for visual QA or as the "
                "'see' step of a computer-use loop."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=E2BDesktopScreenshotBlock.Input,
            output_schema=E2BDesktopScreenshotBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "mock-sandbox-id",
            },
            test_output=[
                (
                    "image",
                    lambda url: isinstance(url, str) and url.startswith("data:image/"),
                ),
            ],
            test_mock={
                "take_screenshot": lambda *args, **kwargs: b"\x89PNG\r\n",
            },
        )

    @staticmethod
    def _take_screenshot(api_key: str, sandbox_id: str) -> bytes:
        desktop = _connect_sandbox(api_key, sandbox_id)
        return bytes(desktop.screenshot())

    async def take_screenshot(self, api_key: str, sandbox_id: str) -> bytes:
        return await asyncio.to_thread(self._take_screenshot, api_key, sandbox_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: "ExecutionContext",
        **kwargs,
    ) -> BlockOutput:
        try:
            image_bytes = await self.take_screenshot(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
            )
            b64 = base64.b64encode(image_bytes).decode()
            stored = await store_media_file(
                file=MediaFileType(f"data:image/png;base64,{b64}"),
                execution_context=execution_context,
                return_format="for_block_output",
            )
            yield "image", stored
        except Exception as e:
            yield "error", str(e)


class E2BDesktopPauseBlock(Block):
    """
    Pause a running E2B Desktop sandbox, preserving its filesystem and memory.

    Pausing stops compute billing while keeping the full sandbox state, so you
    can resume later exactly where you left off — just pass the same sandbox_id
    to any other E2B Desktop block and it resumes automatically. Unlike Kill,
    pausing keeps state (a small storage fee applies) rather than destroying it.

    Note: the live stream drops while paused; restart it with a Create/stream
    step after the sandbox resumes.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description="E2B API key — must match the key used to create the sandbox.",
        )
        sandbox_id: str = SchemaField(
            description="Sandbox ID from the Create Desktop Sandbox block to pause.",
        )

    class Output(BlockSchemaOutput):
        sandbox_id: str = SchemaField(
            description=(
                "ID of the paused sandbox. Pass it to any E2B Desktop block "
                "later to resume the sandbox automatically."
            )
        )
        error: str = SchemaField(description="Error message if the pause failed.")

    def __init__(self):
        super().__init__(
            id="a7b8c9d0-e1f2-3456-abcd-567890123456",
            description=(
                "Pause a running E2B Desktop sandbox to stop compute billing while "
                "keeping its full state. Resume later by passing the sandbox_id to "
                "any other E2B Desktop block."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=E2BDesktopPauseBlock.Input,
            output_schema=E2BDesktopPauseBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "mock-sandbox-id",
            },
            test_output=[
                ("sandbox_id", "mock-sandbox-id"),
            ],
            test_mock={"pause_sandbox": lambda *args, **kwargs: "mock-sandbox-id"},
        )

    @staticmethod
    def _pause_sandbox(api_key: str, sandbox_id: str) -> str:
        desktop = _connect_sandbox(api_key, sandbox_id)
        # pause() returns the (unchanged) sandbox ID; fall back to the input id.
        return desktop.pause() or sandbox_id

    async def pause_sandbox(self, api_key: str, sandbox_id: str) -> str:
        return await asyncio.to_thread(self._pause_sandbox, api_key, sandbox_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            paused_id = await self.pause_sandbox(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
            )
            yield "sandbox_id", paused_id
        except Exception as e:
            yield "error", str(e)


class E2BDesktopKillBlock(Block):
    """
    Destroy a running E2B Desktop sandbox immediately.

    Always call this when you're done — billing stops within seconds. Desktop
    sandboxes that are not killed run until their ``timeout`` expires.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description="E2B API key — must match the key used to create the sandbox.",
        )
        sandbox_id: str = SchemaField(
            description="Sandbox ID from the Create Desktop Sandbox block to destroy.",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the sandbox was destroyed successfully."
        )
        error: str = SchemaField(description="Error message if the kill failed.")

    def __init__(self):
        super().__init__(
            id="e5f6a7b8-c9d0-1234-efab-345678901234",
            description=(
                "Destroy a running E2B Desktop sandbox and stop billing immediately. "
                "Always call this when the desktop is no longer needed."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=E2BDesktopKillBlock.Input,
            output_schema=E2BDesktopKillBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "mock-sandbox-id",
            },
            test_output=[
                ("success", True),
            ],
            test_mock={"kill_sandbox": lambda *args, **kwargs: None},
        )

    @staticmethod
    def _kill_sandbox(api_key: str, sandbox_id: str) -> None:
        desktop = _connect_sandbox(api_key, sandbox_id)
        desktop.kill()

    async def kill_sandbox(self, api_key: str, sandbox_id: str) -> None:
        await asyncio.to_thread(self._kill_sandbox, api_key, sandbox_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            await self.kill_sandbox(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
            )
            yield "success", True
        except Exception as e:
            yield "error", str(e)
            yield "success", False
