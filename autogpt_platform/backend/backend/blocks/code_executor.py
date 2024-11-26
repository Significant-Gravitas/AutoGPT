from enum import Enum
from typing import Literal

from autogpt_libs.supabase_integration_credentials_store.types import APIKeyCredentials
from e2b_code_interpreter import Sandbox
from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsField, CredentialsMetaInput, SchemaField

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="e2b_sandbox",
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


class CodeExecutionBlock(Block):
    # TODO : Add support to upload and download files
    # Currently, You can customized the CPU and Memory, only by creating a pre customized sandbox template
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal["e2b_sandbox"], Literal["api_key"]
        ] = CredentialsField(
            provider="e2b_sandbox",
            supported_credential_types={"api_key"},
            description="Enter your api key for the E2B Sandbox. You can get it in here - https://e2b.dev/docs",
        )

        # Todo : Option to run commond in background
        commands: list[str] = SchemaField(
            description="Run these commands in the sandbox before running the code, You can use `curl` or `git` to install your desired debian based package manager. pip and npm is pre installed\n Currently, none of these commands work in the background.",
            placeholder="pip install cowsay",
            default=[],
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
            description="Sandbox timeout in seconds", default=300
        )

        template_id: str = SchemaField(
            description="Create a pre Customized sandbox template. and Enter the template id here, Check more about it in here - https://e2b.dev/docs/sandbox-template",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        response: str = SchemaField(description="Response from code execution")
        stdout_logs: str = SchemaField(
            description="Standard output logs from execution"
        )
        stderr_logs: str = SchemaField(description="Standard error logs from execution")
        error: str = SchemaField(description="Error message if execution failed")

    def __init__(self):
        super().__init__(
            id="0b02b072-abe7-11ef-8372-fb5d162dd712",
            description="Executes Python code commands in an isolated sandbox environment. Every sandbox has access to internet",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CodeExecutionBlock.Input,
            output_schema=CodeExecutionBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "code": "print('Hello World')",
                "language": ProgrammingLanguage.PYTHON.value,
                "commands": [],
                "timeout": 300,
                "template_id": ""
            },
            test_output=[
                ("response", "Hello World"),
                ("stdout_logs", "Hello World\n"),
                ],
            test_mock={
                "execute_code": lambda code, language, commands, timeout, api_key, template_id: (
                    "Hello World",
                    "Hello World\n",
                    "",
                ),
            },
        )

    def execute_code(
        self,
        code: str,
        language: ProgrammingLanguage,
        commands: list[str],
        timeout: int,
        api_key: str,
        template_id: str,
    ):
        try:
            sandbox = None
            if template_id:
                sandbox = Sandbox(
                    template=template_id, api_key=api_key, timeout=timeout
                )
            else:
                sandbox = Sandbox(api_key=api_key, timeout=timeout)

            if not sandbox:
                raise Exception("Sandbox not created")

            # Running commands
            for cmd in commands:
                sandbox.commands.run(cmd)

            # Executing the code
            execution = sandbox.run_code(
                code,
                language=language.value,
                on_error=lambda e: sandbox.kill(),  # Kill the sandbox if there is an error
            )

            if execution.error:
                raise Exception(execution.error)

            response = execution.text
            stdout_logs = "".join(execution.logs.stdout)
            stderr_logs = "".join(execution.logs.stderr)

            return response, stdout_logs, stderr_logs

        except Exception as e:
            raise e

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            response, stdout_logs, stderr_logs = self.execute_code(
                input_data.code,
                input_data.language,
                input_data.commands,
                input_data.timeout,
                credentials.api_key.get_secret_value(),
                input_data.template_id,
            )

            if response:
                yield "response", response
            if stdout_logs:
                yield "stdout_logs", stdout_logs
            if stderr_logs:
                yield "stderr_logs", stderr_logs
        except Exception as e:
            yield "error", str(e)
