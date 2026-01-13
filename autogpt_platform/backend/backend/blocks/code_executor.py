from enum import Enum
from typing import Any, Literal, Optional

from e2b import AsyncSandbox as BaseAsyncSandbox
from e2b_code_interpreter import AsyncSandbox
from e2b_code_interpreter import Result as E2BExecutionResult
from e2b_code_interpreter.charts import Chart as E2BExecutionResultChart
from pydantic import BaseModel, Field, JsonValue, SecretStr

from backend.data.block import (
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

TEST_ANTHROPIC_CREDENTIALS = APIKeyCredentials(
    id="2e568a2b-b2ea-475a-8564-9a676bf31c56",
    provider="anthropic",
    api_key=SecretStr("mock-anthropic-api-key"),
    title="Mock Anthropic API key",
    expires_at=None,
)
TEST_ANTHROPIC_CREDENTIALS_INPUT = {
    "provider": TEST_ANTHROPIC_CREDENTIALS.provider,
    "id": TEST_ANTHROPIC_CREDENTIALS.id,
    "type": TEST_ANTHROPIC_CREDENTIALS.type,
    "title": TEST_ANTHROPIC_CREDENTIALS.type,
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
    json_data: Optional[JsonValue] = Field(None, alias="json")
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

    class Output(BlockSchemaOutput):
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

    class Output(BlockSchemaOutput):
        sandbox_id: str = SchemaField(description="ID of the sandbox instance")
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

    class Output(BlockSchemaOutput):
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


class ClaudeCodeBlock(Block):
    """
    Execute tasks using Claude Code (Anthropic's AI coding assistant) in an E2B sandbox.

    Claude Code can create files, install tools, run commands, and perform complex
    coding tasks autonomously within a secure sandbox environment.
    """

    # Use base template - we'll install Claude Code ourselves for latest version
    DEFAULT_TEMPLATE = "base"

    class Input(BlockSchemaInput):
        e2b_credentials: CredentialsMetaInput[
            Literal[ProviderName.E2B], Literal["api_key"]
        ] = CredentialsField(
            description=(
                "API key for the E2B platform to create the sandbox. "
                "Get one at https://e2b.dev/docs"
            ),
        )

        anthropic_credentials: CredentialsMetaInput[
            Literal[ProviderName.ANTHROPIC], Literal["api_key"]
        ] = CredentialsField(
            description=(
                "API key for Anthropic to power Claude Code. "
                "Get one at https://console.anthropic.com/"
            ),
        )

        prompt: str = SchemaField(
            description=(
                "The task or instruction for Claude Code to execute. "
                "Claude Code can create files, install packages, run commands, "
                "and perform complex coding tasks."
            ),
            placeholder="Create a hello world index.html file",
            default="",
            advanced=False,
        )

        timeout: int = SchemaField(
            description=(
                "Sandbox timeout in seconds. Claude Code tasks can take "
                "a while, so set this appropriately for your task complexity."
            ),
            default=300,  # 5 minutes default
            advanced=True,
        )

        setup_commands: list[str] = SchemaField(
            description=(
                "Optional shell commands to run before executing Claude Code. "
                "Useful for installing dependencies or setting up the environment."
            ),
            default_factory=list,
            advanced=True,
        )

        working_directory: str = SchemaField(
            description="Working directory for Claude Code to operate in.",
            default="/home/user",
            advanced=True,
        )

        # Session/continuation support
        session_id: str = SchemaField(
            description=(
                "Session ID to resume a previous conversation. "
                "Leave empty for a new conversation. "
                "Use the session_id from a previous run to continue that conversation."
            ),
            default="",
            advanced=True,
        )

        sandbox_id: str = SchemaField(
            description=(
                "Sandbox ID to reconnect to an existing sandbox. "
                "Required when resuming a session (along with session_id). "
                "Use the sandbox_id from a previous run where dispose_sandbox was False."
            ),
            default="",
            advanced=True,
        )

        conversation_history: str = SchemaField(
            description=(
                "Previous conversation history to continue from. "
                "Use this to restore context on a fresh sandbox if the previous one timed out. "
                "Pass the conversation_history output from a previous run."
            ),
            default="",
            advanced=True,
        )

        dispose_sandbox: bool = SchemaField(
            description=(
                "Whether to dispose of the sandbox immediately after execution. "
                "Set to False if you want to continue the conversation later "
                "(you'll need both sandbox_id and session_id from the output)."
            ),
            default=True,
            advanced=True,
        )

    class FileOutput(BaseModel):
        """A file extracted from the sandbox."""

        path: str
        name: str
        content: str

    class Output(BlockSchemaOutput):
        response: str = SchemaField(
            description="The output/response from Claude Code execution"
        )
        files: list["ClaudeCodeBlock.FileOutput"] = SchemaField(
            description=(
                "List of files created/modified by Claude Code. "
                "Each file has 'path', 'name', and 'content' fields."
            )
        )
        conversation_history: str = SchemaField(
            description=(
                "Full conversation history including this turn. "
                "Pass this to conversation_history input to continue on a fresh sandbox "
                "if the previous sandbox timed out."
            )
        )
        session_id: str = SchemaField(
            description=(
                "Session ID for this conversation. "
                "Pass this back along with sandbox_id to continue the conversation."
            )
        )
        sandbox_id: str = SchemaField(
            description=(
                "ID of the sandbox instance. "
                "Pass this back along with session_id to continue the conversation "
                "(only available if dispose_sandbox was False)."
            )
        )

    def __init__(self):
        super().__init__(
            id="4e34f4a5-9b89-4326-ba77-2dd6750b7194",
            description=(
                "Execute tasks using Claude Code in an E2B sandbox. "
                "Claude Code can create files, install tools, run commands, "
                "and perform complex coding tasks autonomously."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.AI},
            input_schema=ClaudeCodeBlock.Input,
            output_schema=ClaudeCodeBlock.Output,
            test_credentials={
                "e2b_credentials": TEST_CREDENTIALS,
                "anthropic_credentials": TEST_ANTHROPIC_CREDENTIALS,
            },
            test_input={
                "e2b_credentials": TEST_CREDENTIALS_INPUT,
                "anthropic_credentials": TEST_ANTHROPIC_CREDENTIALS_INPUT,
                "prompt": "Create a hello world HTML file",
                "timeout": 300,
                "setup_commands": [],
                "working_directory": "/home/user",
                "session_id": "",
                "sandbox_id": "",
                "conversation_history": "",
                "dispose_sandbox": True,
            },
            test_output=[
                ("response", "Created index.html with hello world content"),
                (
                    "files",
                    [
                        {
                            "path": "/home/user/index.html",
                            "name": "index.html",
                            "content": "<html>Hello World</html>",
                        }
                    ],
                ),
                (
                    "conversation_history",
                    "User: Create a hello world HTML file\n"
                    "Claude: Created index.html with hello world content",
                ),
                ("session_id", str),
            ],
            test_mock={
                "execute_claude_code": lambda *args, **kwargs: (
                    "Created index.html with hello world content",  # response
                    [
                        ClaudeCodeBlock.FileOutput(
                            path="/home/user/index.html",
                            name="index.html",
                            content="<html>Hello World</html>",
                        )
                    ],  # files
                    "User: Create a hello world HTML file\n"
                    "Claude: Created index.html with hello world content",  # conversation_history
                    "test-session-id",  # session_id
                    "sandbox_id",  # sandbox_id
                ),
            },
        )

    async def execute_claude_code(
        self,
        e2b_api_key: str,
        anthropic_api_key: str,
        prompt: str,
        timeout: int,
        setup_commands: list[str],
        working_directory: str,
        session_id: str,
        existing_sandbox_id: str,
        conversation_history: str,
        dispose_sandbox: bool,
    ) -> tuple[str, list["ClaudeCodeBlock.FileOutput"], str, str, str]:
        """
        Execute Claude Code in an E2B sandbox.

        Returns:
            Tuple of (response, files, conversation_history, session_id, sandbox_id)
        """
        import json
        import uuid

        sandbox = None

        try:
            # Either reconnect to existing sandbox or create a new one
            if existing_sandbox_id:
                # Reconnect to existing sandbox for conversation continuation
                sandbox = await BaseAsyncSandbox.connect(
                    sandbox_id=existing_sandbox_id,
                    api_key=e2b_api_key,
                )
            else:
                # Create new sandbox
                sandbox = await BaseAsyncSandbox.create(
                    template=self.DEFAULT_TEMPLATE,
                    api_key=e2b_api_key,
                    timeout=timeout,
                    envs={"ANTHROPIC_API_KEY": anthropic_api_key},
                )

                # Install Claude Code from npm (ensures we get the latest version)
                install_result = await sandbox.commands.run(
                    "npm install -g @anthropic-ai/claude-code@latest",
                    timeout=120,  # 2 min timeout for install
                )
                if install_result.exit_code != 0:
                    raise Exception(
                        f"Failed to install Claude Code: {install_result.stderr}"
                    )

                # Run any user-provided setup commands
                for cmd in setup_commands:
                    await sandbox.commands.run(cmd)

            # Generate or use provided session ID
            current_session_id = session_id if session_id else str(uuid.uuid4())

            # Build base Claude flags
            base_flags = "-p --dangerously-skip-permissions --output-format json"

            # Add conversation history context if provided (for fresh sandbox continuation)
            history_flag = ""
            if conversation_history and not session_id:
                # Inject previous conversation as context via system prompt
                # Escape the history for shell
                escaped_history = conversation_history.replace("'", "'\"'\"'")
                history_flag = f" --append-system-prompt 'Previous conversation context: {escaped_history}'"

            # Build Claude command based on whether we're resuming or starting new
            if session_id:
                # Resuming existing session (sandbox still alive)
                claude_command = (
                    f"cd {working_directory} && "
                    f"echo {self._escape_prompt(prompt)} | "
                    f"claude --resume {session_id} {base_flags}"
                )
            else:
                # New session with specific ID
                claude_command = (
                    f"cd {working_directory} && "
                    f"echo {self._escape_prompt(prompt)} | "
                    f"claude --session-id {current_session_id} {base_flags}{history_flag}"
                )

            result = await sandbox.commands.run(
                claude_command,
                timeout=0,  # No command timeout - let sandbox timeout handle it
            )

            raw_output = result.stdout or ""
            sandbox_id = sandbox.sandbox_id

            # Parse JSON output to extract response and build conversation history
            response = ""
            new_conversation_history = conversation_history or ""

            try:
                # The JSON output contains the result
                output_data = json.loads(raw_output)
                response = output_data.get("result", raw_output)

                # Build conversation history entry
                turn_entry = f"User: {prompt}\nClaude: {response}"
                if new_conversation_history:
                    new_conversation_history = (
                        f"{new_conversation_history}\n\n{turn_entry}"
                    )
                else:
                    new_conversation_history = turn_entry

            except json.JSONDecodeError:
                # If not valid JSON, use raw output
                response = raw_output
                turn_entry = f"User: {prompt}\nClaude: {response}"
                if new_conversation_history:
                    new_conversation_history = (
                        f"{new_conversation_history}\n\n{turn_entry}"
                    )
                else:
                    new_conversation_history = turn_entry

            # Extract files from the working directory
            files = await self._extract_files(sandbox, working_directory)

            return (
                response,
                files,
                new_conversation_history,
                current_session_id,
                sandbox_id,
            )

        finally:
            if dispose_sandbox and sandbox:
                await sandbox.kill()

    async def _extract_files(
        self,
        sandbox: BaseAsyncSandbox,
        working_directory: str,
    ) -> list["ClaudeCodeBlock.FileOutput"]:
        """
        Extract files from the sandbox working directory.

        Returns:
            List of FileOutput objects with path, name, and content
        """
        files: list[ClaudeCodeBlock.FileOutput] = []

        # Text file extensions we can safely read
        text_extensions = {
            ".txt",
            ".md",
            ".html",
            ".htm",
            ".css",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".py",
            ".rb",
            ".php",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".swift",
            ".kt",
            ".scala",
            ".sh",
            ".bash",
            ".zsh",
            ".sql",
            ".graphql",
            ".env",
            ".gitignore",
            ".dockerfile",
            "Dockerfile",
            ".vue",
            ".svelte",
            ".astro",
            ".mdx",
            ".rst",
            ".tex",
            ".csv",
            ".log",
        }

        try:
            # List files recursively using find command
            find_result = await sandbox.commands.run(
                f"find {working_directory} -type f -not -path '*/node_modules/*' "
                f"-not -path '*/.git/*' -not -path '*/.*' 2>/dev/null | head -100"
            )

            if find_result.stdout:
                for file_path in find_result.stdout.strip().split("\n"):
                    if not file_path:
                        continue

                    # Check if it's a text file we can read
                    is_text = any(
                        file_path.endswith(ext) for ext in text_extensions
                    ) or file_path.endswith("Dockerfile")

                    if is_text:
                        try:
                            content = await sandbox.files.read(file_path)
                            # Handle bytes or string
                            if isinstance(content, bytes):
                                content = content.decode("utf-8", errors="replace")

                            # Extract filename from path
                            file_name = file_path.split("/")[-1]

                            files.append(
                                ClaudeCodeBlock.FileOutput(
                                    path=file_path,
                                    name=file_name,
                                    content=content,
                                )
                            )
                        except Exception:
                            # Skip files that can't be read
                            pass

        except Exception:
            # If file extraction fails, return empty results
            pass

        return files

    def _escape_prompt(self, prompt: str) -> str:
        """Escape the prompt for safe shell execution."""
        # Use single quotes and escape any single quotes in the prompt
        escaped = prompt.replace("'", "'\"'\"'")
        return f"'{escaped}'"

    async def run(
        self,
        input_data: Input,
        *,
        e2b_credentials: APIKeyCredentials,
        anthropic_credentials: APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            (
                response,
                files,
                conversation_history,
                session_id,
                sandbox_id,
            ) = await self.execute_claude_code(
                e2b_api_key=e2b_credentials.api_key.get_secret_value(),
                anthropic_api_key=anthropic_credentials.api_key.get_secret_value(),
                prompt=input_data.prompt,
                timeout=input_data.timeout,
                setup_commands=input_data.setup_commands,
                working_directory=input_data.working_directory,
                session_id=input_data.session_id,
                existing_sandbox_id=input_data.sandbox_id,
                conversation_history=input_data.conversation_history,
                dispose_sandbox=input_data.dispose_sandbox,
            )

            yield "response", response
            if files:
                # Convert FileOutput objects to dicts for serialization
                yield "files", [f.model_dump() for f in files]
            # Always yield conversation_history so user can restore context on fresh sandbox
            yield "conversation_history", conversation_history
            # Always yield session_id so user can continue conversation
            yield "session_id", session_id
            if not input_data.dispose_sandbox and sandbox_id:
                yield "sandbox_id", sandbox_id

        except Exception as e:
            yield "error", str(e)
