from typing import TYPE_CHECKING, Any, Literal

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.code_executor._base import BaseE2BExecutorMixin
from backend.blocks.code_executor._test import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    mock_execute_code,
)
from backend.blocks.code_executor.helpers import (
    ProgrammingLanguage,
    build_variable_injection,
)
from backend.blocks.code_executor.models import (
    MAIN_RESULT_DESCRIPTION,
    CodeExecutionResult,
    MainCodeExecutionResult,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.sandbox_files import SandboxFileOutput

if TYPE_CHECKING:
    from backend.executor.utils import ExecutionContext


class ExecuteCodeBlock(Block, BaseE2BExecutorMixin):
    # TODO : Add support to upload and download files
    # NOTE: Currently, you can only customize the CPU and Memory
    #       by creating a pre customized sandbox template
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description=(
                "Enter your API key for the E2B platform. "
                "You can get it in here - https://e2b.dev/docs"
            ),
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

        variables: dict[str, Any] = SchemaField(
            title="Variables (Python/JS only)",
            description=(
                "Variables defined here can be used directly in your code. "
                "Each key (`variables_#_{name}`) is injected directly as a local "
                "variable with the same name (`{name}`) in your code. "
                "Values wired in from other blocks keep their type; default values set "
                "on this node come in as strings, so parse them in your code "
                "if you need a number or other type."
            ),
            default_factory=dict,
            advanced=False,
        )

        code: str = SchemaField(
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
            description="Sandbox lifetime in seconds from creation", default=300
        )

        dispose_sandbox: bool = SchemaField(
            description=(
                "Whether to dispose of the sandbox immediately after execution. "
                "If disabled, the sandbox will run until its timeout expires."
            ),
            default=True,
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
        main_result: MainCodeExecutionResult = SchemaField(
            title="Main Result",
            description=MAIN_RESULT_DESCRIPTION,
        )
        results: list[CodeExecutionResult] = SchemaField(
            description="List of results from the code execution"
        )
        response: str = SchemaField(
            title="Main Text Output",
            description="Text output (if any) of the main execution result",
        )
        stdout_logs: str = SchemaField(
            description="Standard output logs from execution"
        )
        stderr_logs: str = SchemaField(description="Standard error logs from execution")
        files: list[SandboxFileOutput] = SchemaField(
            description=(
                "Files created or modified during execution. "
                "Each file has path, name, content, and workspace_ref (if stored)."
            ),
        )

    def __init__(self):
        super().__init__(
            id="0b02b072-abe7-11ef-8372-fb5d162dd712",
            description="Executes code in a sandbox environment with internet access.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=ExecuteCodeBlock.Input,
            output_schema=ExecuteCodeBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "code": "print('Hello World')",
                "language": ProgrammingLanguage.PYTHON.value,
                "setup_commands": [],
                "timeout": 300,
                "template_id": "",
            },
            test_output=[
                ("results", []),
                ("response", "Hello World"),
                ("stdout_logs", "Hello World\n"),
                ("files", []),
            ],
            test_mock={"execute_code": mock_execute_code},
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: "ExecutionContext",
        **kwargs,
    ) -> BlockOutput:
        try:
            # Expose user-provided variables by passing them as a JSON env var and
            # prepending a constant snippet that deserializes them into the runtime.
            # Keeping the data in the env var (not the code string) avoids injection.
            envs, prefix = build_variable_injection(
                input_data.variables, input_data.language
            )
            code = prefix + input_data.code

            results, text_output, stdout, stderr, _, files = await self.execute_code(
                api_key=credentials.api_key.get_secret_value(),
                code=code,
                language=input_data.language,
                template_id=input_data.template_id,
                setup_commands=input_data.setup_commands,
                timeout=input_data.timeout,
                dispose_sandbox=input_data.dispose_sandbox,
                execution_context=execution_context,
                extract_files=True,
                envs=envs,
            )

            # Determine result object shape & filter out empty formats
            main_result, results = self.process_execution_results(results)
            if main_result:
                yield "main_result", main_result
            yield "results", results
            if text_output:
                yield "response", text_output
            if stdout:
                yield "stdout_logs", stdout
            if stderr:
                yield "stderr_logs", stderr
            # Always yield files (empty list if none)
            yield "files", [f.model_dump() for f in files]
        except Exception as e:
            yield "error", str(e)
