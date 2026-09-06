from typing import TYPE_CHECKING, Literal

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.code_executor.desktop import (
    create_desktop_sandbox,
    kill_desktop_sandbox,
)
from backend.blocks.code_executor.helpers import ProgrammingLanguage
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.desktop_preview import create_preview_link

if TYPE_CHECKING:
    pass

from backend.blocks.code_executor._base import BaseE2BExecutorMixin
from backend.blocks.code_executor._test import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    mock_execute_code,
)


class InstantiateCodeSandboxBlock(Block, BaseE2BExecutorMixin):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description=(
                "Enter your API key for the E2B platform. "
                "You can get it in here - https://e2b.dev/docs"
            )
        )

        # Todo : Option to run commond in background
        setup_commands: list[str] = SchemaField(
            description=(
                "Shell commands to set up the sandbox before running the code. "
                "You can use `curl` or `git` to install your desired Debian based "
                "package manager. `pip` and `npm` are pre-installed.\n\n"
                "These commands are executed with `sh`, in the foreground."
            ),
            placeholder="pip install cowsay",
            default_factory=list,
            advanced=False,
        )

        setup_code: str = SchemaField(
            description="Code to execute in the sandbox",
            placeholder="print('Hello, World!')",
            default="",
            advanced=False,
        )

        language: ProgrammingLanguage = SchemaField(
            description="Programming language to execute",
            default=ProgrammingLanguage.PYTHON,
            advanced=False,
        )

        timeout: int = SchemaField(
            description=(
                "Sandbox lifetime in seconds. Choose enough time to run setup and "
                "test through the live URL (up to 3600 seconds)."
            ),
            default=300,
            ge=1,
            le=3600,
        )

        enable_live_view: bool = SchemaField(
            description=(
                "Start an interactive desktop preview and return live_url. Requires a custom "
                "template_id containing both the desktop and code interpreter. "
                "Use the returned sandbox_id with Execute Code Step to work in "
                "the same environment. The preview ends when the sandbox stops."
            ),
            default=False,
            advanced=True,
        )

        template_id: str = SchemaField(
            description=(
                "You can use an E2B sandbox template by entering its ID here. "
                "Check out the E2B docs for more details: "
                "[E2B - Sandbox template](https://e2b.dev/docs/sandbox-template)"
            ),
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        sandbox_id: str = SchemaField(description="ID of the sandbox instance")
        live_url: str = SchemaField(
            description=(
                "Desktop preview link for the signed-in user who created the sandbox. "
                "Returned after setup when live view is enabled. The link expires after 24 hours."
            )
        )
        response: str = SchemaField(
            title="Text Result",
            description="Text result (if any) of the setup code execution",
        )
        stdout_logs: str = SchemaField(
            description="Standard output logs from execution"
        )
        stderr_logs: str = SchemaField(description="Standard error logs from execution")

    def __init__(self):
        super().__init__(
            id="ff0861c9-1726-4aec-9e5b-bf53f3622112",
            description=(
                "Instantiate a sandbox environment with internet access "
                "in which you can execute code with the Execute Code Step block. "
                "Optionally enable a live desktop preview and return its viewing URL."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=InstantiateCodeSandboxBlock.Input,
            output_schema=InstantiateCodeSandboxBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "setup_code": "print('Hello World')",
                "language": ProgrammingLanguage.PYTHON.value,
                "setup_commands": [],
                "timeout": 300,
                "template_id": "",
            },
            test_output=[
                ("sandbox_id", str),
                ("response", "Hello World"),
                ("stdout_logs", "Hello World\n"),
            ],
            test_mock={"execute_code": mock_execute_code},
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        user_id: str = "",
        **kwargs,
    ) -> BlockOutput:
        desktop_id = None
        sandbox_handed_off = False
        live_url = None
        api_key = credentials.api_key.get_secret_value()
        try:
            if input_data.enable_live_view:
                if not user_id:
                    raise ValueError("Live view requires an authenticated user")
                desktop_id, live_url = await create_desktop_sandbox(
                    api_key=api_key,
                    template_id=input_data.template_id,
                    timeout=input_data.timeout,
                )

            _, text_output, stdout, stderr, sandbox_id, _ = await self.execute_code(
                api_key=api_key,
                code=input_data.setup_code,
                language=input_data.language,
                template_id=input_data.template_id,
                setup_commands=input_data.setup_commands,
                timeout=input_data.timeout,
                sandbox_id=desktop_id,
            )
            if not sandbox_id:
                yield "error", "Sandbox ID not found"
                return
            if live_url:
                yield "live_url", create_preview_link(user_id, live_url)
            sandbox_handed_off = True
            yield "sandbox_id", sandbox_id

            if text_output:
                yield "response", text_output
            if stdout:
                yield "stdout_logs", stdout
            if stderr:
                yield "stderr_logs", stderr
        except Exception as e:
            yield "error", str(e)
        finally:
            if desktop_id and not sandbox_handed_off:
                await kill_desktop_sandbox(api_key, desktop_id)
