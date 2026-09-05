from typing import TYPE_CHECKING, Literal, Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.code_executor.helpers import ProgrammingLanguage
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

if TYPE_CHECKING:
    pass

from backend.blocks.code_executor._base import BaseE2BExecutorMixin
from backend.blocks.code_executor._test import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    mock_execute_code,
)
from backend.blocks.code_executor.models import (
    MAIN_RESULT_DESCRIPTION,
    CodeExecutionResult,
    MainCodeExecutionResult,
)


class ExecuteCodeStepBlock(Block, BaseE2BExecutorMixin):
    class Input(BlockSchemaInput):
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

        timeout: Optional[int] = SchemaField(
            description=(
                "Extend the remaining sandbox lifetime to at least this many seconds "
                "when connecting (up to 3600). Cannot shorten a longer remaining lifetime. "
                "If omitted, E2B's default extension applies. Use dispose_sandbox to stop early."
            ),
            default=None,
            ge=1,
            le=3600,
            advanced=False,
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
            test_mock={"execute_code": mock_execute_code},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            results, text_output, stdout, stderr, _, _ = await self.execute_code(
                api_key=credentials.api_key.get_secret_value(),
                code=input_data.step_code,
                language=input_data.language,
                sandbox_id=input_data.sandbox_id,
                dispose_sandbox=input_data.dispose_sandbox,
                timeout=input_data.timeout,
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
