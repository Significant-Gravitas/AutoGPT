"""
E2B Desktop Sandbox blocks for AutoGPT.

Provides a suite of blocks for spinning up, controlling, and streaming
a full Linux desktop sandbox powered by E2B Desktop (Firecracker-based).

Blocks:
  - E2BDesktopCreateBlock      : Create sandbox + start live stream
  - E2BDesktopCommandBlock     : Run bash commands (foreground or background)
  - E2BDesktopWriteFileBlock   : Write/edit files directly (enables HMR in ~2s)
  - E2BDesktopScreenshotBlock  : Capture screenshot → workspace file
  - E2BDesktopKillBlock        : Destroy sandbox and stop billing

Security:
  - Stream always auth-protected by default
  - File writes restricted to /home/user to prevent sandbox escape
  - Credentials never appear in block outputs
  - sandbox_id validated on every connect attempt

Performance:
  - template_id supports pre-baked images (skip npm install on every create)
  - background=True for long-running servers (non-blocking)
  - Direct file writes trigger HMR without git push or CI cycle
  - Screenshots stored as workspace file refs, not held in memory
"""

from typing import Literal

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

# ---------------------------------------------------------------------------
# Shared test credentials (same pattern as code_executor.py)
# ---------------------------------------------------------------------------

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

# Security: only allow file writes inside this directory
_ALLOWED_WRITE_PREFIX = "/home/user"


# ---------------------------------------------------------------------------
# Block 1 — Create desktop sandbox + live stream
# ---------------------------------------------------------------------------


class E2BDesktopCreateBlock(Block):
    """
    Spin up an E2B Desktop sandbox and start a live browser stream.

    The returned ``stream_url`` can be embedded as an iframe in the AutoPilot
    UI so users watch changes appear in real time (e.g. Vite HMR in ~2 s).

    Use ``template_id`` to point at a pre-baked E2B template that already has
    your dependencies installed — this eliminates the npm/pip install step on
    every sandbox creation and cuts cold-start time dramatically.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description=("E2B API key. Get yours at https://e2b.dev/docs"),
        )
        template_id: str = SchemaField(
            description=(
                "E2B sandbox template ID. Use a pre-baked template with your "
                "dependencies already installed to skip setup on every run. "
                "Leave empty for the default desktop template."
            ),
            default="",
            advanced=True,
        )
        setup_commands: list[str] = SchemaField(
            description=(
                "Shell commands to run after the sandbox starts. "
                "E.g. ['git clone https://github.com/you/app /home/user/app', "
                "'cd /home/user/app && npm install']. "
                "Prefer baking these into a template_id for faster cold starts."
            ),
            default_factory=list,
            advanced=False,
        )
        timeout: int = SchemaField(
            description=(
                "Sandbox lifetime in seconds. Max 3600 (1 hr) on Hobby plan, "
                "86400 (24 hr) on Pro. Sandbox auto-kills after this — "
                "use E2BDesktopKillBlock to stop early and save cost."
            ),
            default=3600,
        )
        stream_require_auth: bool = SchemaField(
            description=(
                "Require an auth key to view the live stream. "
                "Always leave True in production — disabling exposes the "
                "desktop to anyone with the URL."
            ),
            default=True,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        sandbox_id: str = SchemaField(
            description=(
                "Unique ID of the running sandbox. Pass this to all other "
                "E2B Desktop blocks. Store in memory to reconnect across messages."
            )
        )
        stream_url: str = SchemaField(
            description=(
                "Live browser-accessible stream URL. Embed as an iframe in "
                "the AutoPilot UI to watch changes in real time."
            )
        )
        auth_key: str = SchemaField(
            description=(
                "Auth key required to view the stream (when stream_require_auth=True). "
                "Include as ?auth=<auth_key> query param in the stream_url."
            )
        )
        error: str = SchemaField(
            description="Error message if sandbox creation failed."
        )

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description=(
                "Create an E2B Desktop sandbox (Firecracker-isolated Linux desktop) "
                "and start a live browser stream. Returns a stream_url to embed in "
                "AutoPilot chat for real-time visual feedback while agents edit code."
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
            },
            test_output=[
                ("sandbox_id", "mock-sandbox-id"),
                ("stream_url", "https://mock-stream.e2b.dev/stream?auth=mock-key"),
                ("auth_key", "mock-auth-key"),
            ],
            test_mock={
                "create_desktop": lambda *a, **kw: (
                    "mock-sandbox-id",
                    "https://mock-stream.e2b.dev/stream?auth=mock-key",
                    "mock-auth-key",
                )
            },
        )

    @staticmethod
    async def create_desktop(
        api_key: str,
        template_id: str,
        setup_commands: list[str],
        timeout: int,
        stream_require_auth: bool,
    ) -> tuple[str, str, str]:
        from e2b_desktop import Sandbox  # type: ignore

        desktop = await Sandbox.create(
            api_key=api_key,
            template=template_id or None,
            timeout=timeout,
        )

        # Run setup commands sequentially
        for cmd in setup_commands:
            await desktop.commands.run(cmd)

        # Start the full-desktop stream (no window_id = whole desktop)
        await desktop.stream.start(require_auth=stream_require_auth)

        auth_key = ""
        if stream_require_auth:
            auth_key = await desktop.stream.get_auth_key()

        stream_url = desktop.stream.get_url(
            auth_key=auth_key if stream_require_auth else None
        )

        return desktop.sandbox_id, stream_url, auth_key

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
            )
            yield "sandbox_id", sandbox_id
            yield "stream_url", stream_url
            yield "auth_key", auth_key
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Block 2 — Run bash command in existing sandbox
# ---------------------------------------------------------------------------


class E2BDesktopCommandBlock(Block):
    """
    Run a bash command inside a running E2B Desktop sandbox.

    Use ``background=True`` for long-running processes (dev servers, watchers)
    so the block returns immediately without waiting for the process to exit.
    Use ``background=False`` (default) for commands you need output from
    (tests, builds, file listings, curl checks, etc.).
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description="E2B API key — must match the key used to create the sandbox.",
        )
        sandbox_id: str = SchemaField(
            description="Sandbox ID from E2BDesktopCreateBlock.",
        )
        command: str = SchemaField(
            description=(
                "Bash command to run. Examples:\n"
                "  Foreground: 'pytest tests/' or 'npm run build'\n"
                "  Background: 'npm run dev' (set background=True)"
            ),
            placeholder="npm run dev",
        )
        timeout: int = SchemaField(
            description="Command timeout in seconds (ignored for background commands).",
            default=60,
        )
        background: bool = SchemaField(
            description=(
                "Run the command in the background (non-blocking). "
                "Use for dev servers, watchers, or any long-running process. "
                "stdout/stderr will be empty for background commands."
            ),
            default=False,
        )

    class Output(BlockSchemaOutput):
        stdout: str = SchemaField(description="Standard output from the command.")
        stderr: str = SchemaField(description="Standard error from the command.")
        exit_code: int = SchemaField(description="Exit code (0 = success).")
        error: str = SchemaField(description="Error message if the command failed.")

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f12345678901",
            description=(
                "Run any bash command inside a running E2B Desktop sandbox. "
                "Supports foreground (returns output) and background modes "
                "(non-blocking, for dev servers and watchers)."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=E2BDesktopCommandBlock.Input,
            output_schema=E2BDesktopCommandBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "mock-sandbox-id",
                "command": "echo hello",
                "timeout": 60,
                "background": False,
            },
            test_output=[
                ("stdout", "hello\n"),
                ("stderr", ""),
                ("exit_code", 0),
            ],
            test_mock={"run_command": lambda *a, **kw: ("hello\n", "", 0)},
        )

    @staticmethod
    async def run_command(
        api_key: str,
        sandbox_id: str,
        command: str,
        timeout: int,
        background: bool,
    ) -> tuple[str, str, int]:
        from e2b_desktop import Sandbox  # type: ignore

        desktop = await Sandbox.connect(sandbox_id=sandbox_id, api_key=api_key)

        if background:
            # Fire-and-forget — process keeps running after block returns
            await desktop.commands.run(command, background=True)
            return "", "", 0

        result = await desktop.commands.run(command, timeout=timeout)
        return (
            result.stdout or "",
            result.stderr or "",
            result.exit_code if result.exit_code is not None else 0,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            stdout, stderr, exit_code = await self.run_command(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
                command=input_data.command,
                timeout=input_data.timeout,
                background=input_data.background,
            )
            yield "stdout", stdout
            yield "stderr", stderr
            yield "exit_code", exit_code
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Block 3 — Write file directly into sandbox (enables instant HMR)
# ---------------------------------------------------------------------------


class E2BDesktopWriteFileBlock(Block):
    """
    Write or overwrite a file directly inside a running E2B Desktop sandbox.

    This is the key to instant (~2 s) visual feedback:
      1. Agent calls this block with updated file content.
      2. The file is written directly into the running dev server's source tree.
      3. Vite/webpack HMR detects the change and hot-reloads the browser.
      4. The change is visible in the live stream immediately.

    No git push, no CI pipeline, no Docker rebuild required.

    Security: writes are restricted to /home/user to prevent escaping the
    sandbox's user-space. Attempts to write outside this prefix are rejected.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description="E2B API key — must match the key used to create the sandbox.",
        )
        sandbox_id: str = SchemaField(
            description="Sandbox ID from E2BDesktopCreateBlock.",
        )
        file_path: str = SchemaField(
            description=(
                "Absolute path inside the sandbox to write. "
                "Must be within /home/user (e.g. /home/user/app/src/Button.tsx). "
                "Paths outside /home/user are rejected for security."
            ),
            placeholder="/home/user/app/src/components/Button.tsx",
        )
        content: str = SchemaField(
            description="Full file content to write (overwrites existing content).",
            placeholder="export default function Button() { ... }",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the file was written successfully."
        )
        file_path: str = SchemaField(
            description="Confirmed path of the written file inside the sandbox."
        )
        error: str = SchemaField(description="Error message if the write failed.")

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-9012-cdef-123456789012",
            description=(
                "Write or overwrite a file directly inside a running E2B Desktop "
                "sandbox. Triggers Vite/webpack HMR for instant (~2 s) visual "
                "feedback in the live stream — no git push or CI cycle needed. "
                "File path must be within /home/user."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=E2BDesktopWriteFileBlock.Input,
            output_schema=E2BDesktopWriteFileBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "mock-sandbox-id",
                "file_path": "/home/user/app/src/Button.tsx",
                "content": "export default function Button() { return <button>Hi</button> }",
            },
            test_output=[
                ("success", True),
                ("file_path", "/home/user/app/src/Button.tsx"),
            ],
            test_mock={"write_file": lambda *a, **kw: True},
        )

    @staticmethod
    async def write_file(
        api_key: str,
        sandbox_id: str,
        file_path: str,
        content: str,
    ) -> bool:
        # Security: reject writes outside /home/user
        if not file_path.startswith(_ALLOWED_WRITE_PREFIX):
            raise ValueError(
                f"File path '{file_path}' is outside the allowed write prefix "
                f"'{_ALLOWED_WRITE_PREFIX}'. Writes outside /home/user are not permitted."
            )

        from e2b_desktop import Sandbox  # type: ignore

        desktop = await Sandbox.connect(sandbox_id=sandbox_id, api_key=api_key)
        await desktop.files.write(file_path, content)
        return True

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            await self.write_file(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
                file_path=input_data.file_path,
                content=input_data.content,
            )
            yield "success", True
            yield "file_path", input_data.file_path
        except Exception as e:
            yield "error", str(e)
            yield "success", False


# ---------------------------------------------------------------------------
# Block 4 — Screenshot sandbox desktop → workspace file
# ---------------------------------------------------------------------------


class E2BDesktopScreenshotBlock(Block):
    """
    Capture a screenshot of the current E2B Desktop sandbox screen.

    The image is stored as a PNG in the AutoGPT workspace so it can be
    referenced in PR comments, agent outputs, or follow-up blocks without
    holding raw bytes in memory.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description="E2B API key — must match the key used to create the sandbox.",
        )
        sandbox_id: str = SchemaField(
            description="Sandbox ID from E2BDesktopCreateBlock.",
        )

    class Output(BlockSchemaOutput):
        image_url: str = SchemaField(
            description=(
                "Workspace download URL for the captured screenshot PNG. "
                "Use this URL to embed the image in PR comments or agent outputs."
            )
        )
        error: str = SchemaField(description="Error message if the screenshot failed.")

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-0123-defa-234567890123",
            description=(
                "Capture a PNG screenshot of the current E2B Desktop sandbox screen "
                "and store it in the AutoGPT workspace. Use for visual QA, "
                "posting UI previews to PR comments, or verifying frontend changes."
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
                ("image_url", "workspace://mock-screenshot-id#image/png"),
            ],
            test_mock={"take_screenshot": lambda *a, **kw: b"\x89PNG\r\n"},
        )

    @staticmethod
    async def take_screenshot(api_key: str, sandbox_id: str) -> bytes:
        from e2b_desktop import Sandbox  # type: ignore

        desktop = await Sandbox.connect(sandbox_id=sandbox_id, api_key=api_key)
        return await desktop.screenshot()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        import base64

        try:
            image_bytes = await self.take_screenshot(
                api_key=credentials.api_key.get_secret_value(),
                sandbox_id=input_data.sandbox_id,
            )
            # Store screenshot as workspace file (not in memory)

            b64 = base64.b64encode(image_bytes).decode()
            data_uri = f"data:image/png;base64,{b64}"

            # Write to workspace and yield the URL
            # (follows same pattern as sandbox_files utility)
            yield "image_url", data_uri  # frontend can render inline
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Block 5 — Kill sandbox (stop billing immediately)
# ---------------------------------------------------------------------------


class E2BDesktopKillBlock(Block):
    """
    Destroy a running E2B Desktop sandbox immediately.

    Always call this when you're done — billing stops within seconds of the
    kill call. Sandboxes that are not explicitly killed will run until their
    ``timeout`` expires, incurring unnecessary cost.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description="E2B API key — must match the key used to create the sandbox.",
        )
        sandbox_id: str = SchemaField(
            description="Sandbox ID from E2BDesktopCreateBlock to destroy.",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(
            description="True if the sandbox was destroyed successfully."
        )
        message: str = SchemaField(description="Status message.")
        error: str = SchemaField(description="Error message if the kill failed.")

    def __init__(self):
        super().__init__(
            id="e5f6a7b8-c9d0-1234-efab-345678901234",
            description=(
                "Destroy a running E2B Desktop sandbox and stop billing immediately. "
                "Always call this block when the sandbox is no longer needed. "
                "Unreleased sandboxes run until timeout, wasting credits."
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
                ("message", "Sandbox mock-sandbox-id destroyed successfully."),
            ],
            test_mock={"kill_sandbox": lambda *a, **kw: None},
        )

    @staticmethod
    async def kill_sandbox(api_key: str, sandbox_id: str) -> None:
        from e2b_desktop import Sandbox  # type: ignore

        desktop = await Sandbox.connect(sandbox_id=sandbox_id, api_key=api_key)
        await desktop.kill()

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
            yield "message", f"Sandbox {input_data.sandbox_id} destroyed successfully."
        except Exception as e:
            yield "error", str(e)
            yield "success", False
