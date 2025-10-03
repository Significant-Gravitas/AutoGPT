from enum import Enum
from typing import Any, Literal, Optional

from e2b_code_interpreter import AsyncSandbox
from e2b_code_interpreter import Result as E2BExecutionResult
from e2b_code_interpreter.charts import Chart as E2BExecutionResultChart
from pydantic import BaseModel, JsonValue, SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
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


class ProgrammingLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "js"
    BASH = "bash"
    R = "r"
    JAVA = "java"


class MainCodeExecutionResult(BaseModel):
    """
    *Pydantic model mirroring `e2b_code_interpreter.Result`*

    Represents the data to be displayed as a result of executing a cell in a Jupyter notebook.
    The result is similar to the structure returned by ipython kernel: https://ipython.readthedocs.io/en/stable/development/execution.html#execution-semantics

    The result can contain multiple types of data, such as text, images, plots, etc. Each type of data is represented
    as a string, and the result can contain multiple types of data. The display calls don't have to have text representation,
    for the actual result the representation is always present for the result, the other representations are always optional.
    """  # noqa

    class Chart(BaseModel, E2BExecutionResultChart):
        pass

    text: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    svg: Optional[str] = None
    png: Optional[str] = None
    jpeg: Optional[str] = None
    pdf: Optional[str] = None
    latex: Optional[str] = None
    json: Optional[JsonValue] = None  # type: ignore (reportIncompatibleMethodOverride)
    javascript: Optional[str] = None
    data: Optional[dict] = None
    chart: Optional[Chart] = None
    extra: Optional[dict] = None
    """Extra data that can be included. Not part of the standard types."""


class CodeExecutionResult(MainCodeExecutionResult):
    __doc__ = MainCodeExecutionResult.__doc__

    is_main_result: bool = False
    """Whether this data is the main result of the cell. Data can be produced by display calls of which can be multiple in a cell."""  # noqa


class BaseE2BExecutorMixin:
    """Shared implementation methods for E2B executor blocks."""

    async def execute_code(
        self,
        api_key: str,
        code: str,
        language: ProgrammingLanguage,
        template_id: str = "",
        setup_commands: Optional[list[str]] = None,
        timeout: Optional[int] = None,
        sandbox_id: Optional[str] = None,
        dispose_sandbox: bool = False,
    ):
        """
        Unified code execution method that handles all three use cases:
        1. Create new sandbox and execute (ExecuteCodeBlock)
        2. Create new sandbox, execute, and return sandbox_id (InstantiateCodeSandboxBlock)
        3. Connect to existing sandbox and execute (ExecuteCodeStepBlock)
        """  # noqa
        sandbox = None
        try:
            if sandbox_id:
                # Connect to existing sandbox (ExecuteCodeStepBlock case)
                sandbox = await AsyncSandbox.connect(
                    sandbox_id=sandbox_id, api_key=api_key
                )
            else:
                # Create new sandbox (ExecuteCodeBlock/InstantiateCodeSandboxBlock case)
                sandbox = await AsyncSandbox.create(
                    api_key=api_key, template=template_id, timeout=timeout
                )
                if setup_commands:
                    for cmd in setup_commands:
                        await sandbox.commands.run(cmd)

            # Execute the code
            execution = await sandbox.run_code(
                code,
                language=language.value,
                on_error=lambda e: sandbox.kill(),  # Kill the sandbox on error
            )

            if execution.error:
                raise Exception(execution.error)

            results = execution.results
            text_output = execution.text
            stdout_logs = "".join(execution.logs.stdout)
            stderr_logs = "".join(execution.logs.stderr)

            return results, text_output, stdout_logs, stderr_logs, sandbox.sandbox_id
        finally:
            # Dispose of sandbox if requested to reduce usage costs
            if dispose_sandbox and sandbox:
                await sandbox.kill()

    def process_execution_results(
        self, results: list[E2BExecutionResult]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Process and filter execution results."""
        # Filter out empty formats and convert to dicts
        processed_results = [
            {
                f: value
                for f in [*r.formats(), "extra", "is_main_result"]
                if (value := getattr(r, f, None)) is not None
            }
            for r in results
        ]
        if main_result := next(
            (r for r in processed_results if r.get("is_main_result")), None
        ):
            # Make main_result a copy we can modify & remove is_main_result
            (main_result := {**main_result}).pop("is_main_result")

        return main_result, processed_results


class ExecuteCodeBlock(Block, BaseE2BExecutorMixin):
    # TODO : Add support to upload and download files
    # NOTE: Currently, you can only customize the CPU and Memory
    #       by creating a pre customized sandbox template
    class Input(BlockSchema):
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
            description="Execution timeout in seconds", default=300
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

    class Output(BlockSchema):
        main_result: MainCodeExecutionResult = SchemaField(
            title="Main Result", description="The main result from the code execution"
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
        error: str = SchemaField(description="Error message if execution failed")

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
            ],
            test_mock={
                "execute_code": lambda api_key, code, language, template_id, setup_commands, timeout, dispose_sandbox: (  # noqa
                    [],  # results
                    "Hello World",  # text_output
                    "Hello World\n",  # stdout_logs
                    "",  # stderr_logs
                    "sandbox_id",  # sandbox_id
                ),
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            results, text_output, stdout, stderr, _ = await self.execute_code(
                api_key=credentials.api_key.get_secret_value(),
                code=input_data.code,
                language=input_data.language,
                template_id=input_data.template_id,
                setup_commands=input_data.setup_commands,
                timeout=input_data.timeout,
                dispose_sandbox=input_data.dispose_sandbox,
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
        except Exception as e:
            yield "error", str(e)


class InstantiateCodeSandboxBlock(Block, BaseE2BExecutorMixin):
    class Input(BlockSchema):
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
            description="Execution timeout in seconds", default=300
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

    class Output(BlockSchema):
        sandbox_id: str = SchemaField(description="ID of the sandbox instance")
        response: str = SchemaField(
            title="Text Result",
            description="Text result (if any) of the setup code execution",
        )
        stdout_logs: str = SchemaField(
            description="Standard output logs from execution"
        )
        stderr_logs: str = SchemaField(description="Standard error logs from execution")
        error: str = SchemaField(description="Error message if execution failed")

    def __init__(self):
        super().__init__(
            id="ff0861c9-1726-4aec-9e5b-bf53f3622112",
            description=(
                "Instantiate a sandbox environment with internet access "
                "in which you can execute code with the Execute Code Step block."
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
            test_mock={
                "execute_code": lambda api_key, code, language, template_id, setup_commands, timeout: (  # noqa
                    [],  # results
                    "Hello World",  # text_output
                    "Hello World\n",  # stdout_logs
                    "",  # stderr_logs
                    "sandbox_id",  # sandbox_id
                ),
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            _, text_output, stdout, stderr, sandbox_id = await self.execute_code(
                api_key=credentials.api_key.get_secret_value(),
                code=input_data.setup_code,
                language=input_data.language,
                template_id=input_data.template_id,
                setup_commands=input_data.setup_commands,
                timeout=input_data.timeout,
            )
            if sandbox_id:
                yield "sandbox_id", sandbox_id
            else:
                yield "error", "Sandbox ID not found"

            if text_output:
                yield "response", text_output
            if stdout:
                yield "stdout_logs", stdout
            if stderr:
                yield "stderr_logs", stderr
        except Exception as e:
            yield "error", str(e)


class ExecuteCodeStepBlock(Block, BaseE2BExecutorMixin):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description=(
                "Enter your API key for the E2B platform. "
                "You can get it in here - https://e2b.dev/docs"
            ),
        )

        sandbox_id: str = SchemaField(
            description="ID of the sandbox instance to execute the code in",
            advanced=False,
        )

        step_code: str = SchemaField(
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

        dispose_sandbox: bool = SchemaField(
            description="Whether to dispose of the sandbox after executing this code.",
            default=False,
        )

    class Output(BlockSchema):
        main_result: MainCodeExecutionResult = SchemaField(
            title="Main Result", description="The main result from the code execution"
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
        error: str = SchemaField(description="Error message if execution failed")

    def __init__(self):
        super().__init__(
            id="82b59b8e-ea10-4d57-9161-8b169b0adba6",
            description="Execute code in a previously instantiated sandbox.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=ExecuteCodeStepBlock.Input,
            output_schema=ExecuteCodeStepBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "sandbox_id": "sandbox_id",
                "step_code": "print('Hello World')",
                "language": ProgrammingLanguage.PYTHON.value,
            },
            test_output=[
                ("results", []),
                ("response", "Hello World"),
                ("stdout_logs", "Hello World\n"),
            ],
            test_mock={
                "execute_code": lambda api_key, code, language, sandbox_id, dispose_sandbox: (  # noqa
                    [],  # results
                    "Hello World",  # text_output
                    "Hello World\n",  # stdout_logs
                    "",  # stderr_logs
                    sandbox_id,  # sandbox_id
                ),
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            results, text_output, stdout, stderr, _ = await self.execute_code(
                api_key=credentials.api_key.get_secret_value(),
                code=input_data.step_code,
                language=input_data.language,
                sandbox_id=input_data.sandbox_id,
                dispose_sandbox=input_data.dispose_sandbox,
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
        except Exception as e:
            yield "error", str(e)
