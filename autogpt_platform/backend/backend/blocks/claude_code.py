import json
import shlex
import uuid
from typing import Literal, Optional

from e2b import AsyncSandbox as BaseAsyncSandbox
from pydantic import BaseModel, SecretStr

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


class ClaudeCodeExecutionError(Exception):
    """Exception raised when Claude Code execution fails.

    Carries the sandbox_id so it can be returned to the user for cleanup
    when dispose_sandbox=False.
    """

    def __init__(self, message: str, sandbox_id: str = ""):
        super().__init__(message)
        self.sandbox_id = sandbox_id


# Test credentials for E2B
TEST_E2B_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="e2b",
    api_key=SecretStr("mock-e2b-api-key"),
    title="Mock E2B API key",
    expires_at=None,
)
TEST_E2B_CREDENTIALS_INPUT = {
    "provider": TEST_E2B_CREDENTIALS.provider,
    "id": TEST_E2B_CREDENTIALS.id,
    "type": TEST_E2B_CREDENTIALS.type,
    "title": TEST_E2B_CREDENTIALS.title,
}

# Test credentials for Anthropic
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
    "title": TEST_ANTHROPIC_CREDENTIALS.title,
}


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
                "Get one on the [e2b website](https://e2b.dev/docs)"
            ),
        )

        anthropic_credentials: CredentialsMetaInput[
            Literal[ProviderName.ANTHROPIC], Literal["api_key"]
        ] = CredentialsField(
            description=(
                "API key for Anthropic to power Claude Code. "
                "Get one at [Anthropic's website](https://console.anthropic.com)"
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
                "a while, so set this appropriately for your task complexity. "
                "Note: This only applies when creating a new sandbox. "
                "When reconnecting to an existing sandbox via sandbox_id, "
                "the original timeout is retained."
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
        relative_path: str  # Path relative to working directory (for GitHub, etc.)
        name: str
        content: str

    class Output(BlockSchemaOutput):
        response: str = SchemaField(
            description="The output/response from Claude Code execution"
        )
        files: list["ClaudeCodeBlock.FileOutput"] = SchemaField(
            description=(
                "List of text files created/modified by Claude Code during this execution. "
                "Each file has 'path', 'relative_path', 'name', and 'content' fields."
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
        sandbox_id: Optional[str] = SchemaField(
            description=(
                "ID of the sandbox instance. "
                "Pass this back along with session_id to continue the conversation. "
                "This is None if dispose_sandbox was True (sandbox was disposed)."
            ),
            default=None,
        )
        error: str = SchemaField(description="Error message if execution failed")

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
                "e2b_credentials": TEST_E2B_CREDENTIALS,
                "anthropic_credentials": TEST_ANTHROPIC_CREDENTIALS,
            },
            test_input={
                "e2b_credentials": TEST_E2B_CREDENTIALS_INPUT,
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
                            "relative_path": "index.html",
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
                ("sandbox_id", None),  # None because dispose_sandbox=True in test_input
            ],
            test_mock={
                "execute_claude_code": lambda *args, **kwargs: (
                    "Created index.html with hello world content",  # response
                    [
                        ClaudeCodeBlock.FileOutput(
                            path="/home/user/index.html",
                            relative_path="index.html",
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

        # Validate that sandbox_id is provided when resuming a session
        if session_id and not existing_sandbox_id:
            raise ValueError(
                "sandbox_id is required when resuming a session with session_id. "
                "The session state is stored in the original sandbox. "
                "If the sandbox has timed out, use conversation_history instead "
                "to restore context on a fresh sandbox."
            )

        sandbox = None
        sandbox_id = ""

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
                    setup_result = await sandbox.commands.run(cmd)
                    if setup_result.exit_code != 0:
                        raise Exception(
                            f"Setup command failed: {cmd}\n"
                            f"Exit code: {setup_result.exit_code}\n"
                            f"Stdout: {setup_result.stdout}\n"
                            f"Stderr: {setup_result.stderr}"
                        )

            # Capture sandbox_id immediately after creation/connection
            # so it's available for error recovery if dispose_sandbox=False
            sandbox_id = sandbox.sandbox_id

            # Generate or use provided session ID
            current_session_id = session_id if session_id else str(uuid.uuid4())

            # Build base Claude flags
            base_flags = "-p --dangerously-skip-permissions --output-format json"

            # Add conversation history context if provided (for fresh sandbox continuation)
            history_flag = ""
            if conversation_history and not session_id:
                # Inject previous conversation as context via system prompt
                # Use consistent escaping via _escape_prompt helper
                escaped_history = self._escape_prompt(
                    f"Previous conversation context: {conversation_history}"
                )
                history_flag = f" --append-system-prompt {escaped_history}"

            # Build Claude command based on whether we're resuming or starting new
            # Use shlex.quote for working_directory and session IDs to prevent injection
            safe_working_dir = shlex.quote(working_directory)
            if session_id:
                # Resuming existing session (sandbox still alive)
                safe_session_id = shlex.quote(session_id)
                claude_command = (
                    f"cd {safe_working_dir} && "
                    f"echo {self._escape_prompt(prompt)} | "
                    f"claude --resume {safe_session_id} {base_flags}"
                )
            else:
                # New session with specific ID
                safe_current_session_id = shlex.quote(current_session_id)
                claude_command = (
                    f"cd {safe_working_dir} && "
                    f"echo {self._escape_prompt(prompt)} | "
                    f"claude --session-id {safe_current_session_id} {base_flags}{history_flag}"
                )

            # Capture timestamp before running Claude Code to filter files later
            # Capture timestamp 1 second in the past to avoid race condition with file creation
            timestamp_result = await sandbox.commands.run(
                "date -u -d '1 second ago' +%Y-%m-%dT%H:%M:%S"
            )
            if timestamp_result.exit_code != 0:
                raise RuntimeError(
                    f"Failed to capture timestamp: {timestamp_result.stderr}"
                )
            start_timestamp = (
                timestamp_result.stdout.strip() if timestamp_result.stdout else None
            )

            result = await sandbox.commands.run(
                claude_command,
                timeout=0,  # No command timeout - let sandbox timeout handle it
            )

            # Check for command failure
            if result.exit_code != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise Exception(
                    f"Claude Code command failed with exit code {result.exit_code}:\n"
                    f"{error_msg}"
                )

            raw_output = result.stdout or ""

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

            # Extract files created/modified during this run
            files = await self._extract_files(
                sandbox, working_directory, start_timestamp
            )

            return (
                response,
                files,
                new_conversation_history,
                current_session_id,
                sandbox_id,
            )

        except Exception as e:
            # Wrap exception with sandbox_id so caller can access/cleanup
            # the preserved sandbox when dispose_sandbox=False
            raise ClaudeCodeExecutionError(str(e), sandbox_id) from e

        finally:
            if dispose_sandbox and sandbox:
                await sandbox.kill()

    async def _extract_files(
        self,
        sandbox: BaseAsyncSandbox,
        working_directory: str,
        since_timestamp: str | None = None,
    ) -> list["ClaudeCodeBlock.FileOutput"]:
        """
        Extract text files created/modified during this Claude Code execution.

        Args:
            sandbox: The E2B sandbox instance
            working_directory: Directory to search for files
            since_timestamp: ISO timestamp - only return files modified after this time

        Returns:
            List of FileOutput objects with path, relative_path, name, and content
        """
        files: list[ClaudeCodeBlock.FileOutput] = []

        # Text file extensions we can safely read as text
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
            # Exclude node_modules and .git directories, but allow hidden files
            # like .env and .gitignore (they're filtered by text_extensions later)
            # Filter by timestamp to only get files created/modified during this run
            safe_working_dir = shlex.quote(working_directory)
            timestamp_filter = ""
            if since_timestamp:
                timestamp_filter = f"-newermt {shlex.quote(since_timestamp)} "
            find_result = await sandbox.commands.run(
                f"find {safe_working_dir} -type f "
                f"{timestamp_filter}"
                f"-not -path '*/node_modules/*' "
                f"-not -path '*/.git/*' "
                f"2>/dev/null"
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

                            # Calculate relative path by stripping working directory
                            relative_path = file_path
                            if file_path.startswith(working_directory):
                                relative_path = file_path[len(working_directory) :]
                                # Remove leading slash if present
                                if relative_path.startswith("/"):
                                    relative_path = relative_path[1:]

                            files.append(
                                ClaudeCodeBlock.FileOutput(
                                    path=file_path,
                                    relative_path=relative_path,
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
            # Always yield files (empty list if none) to match Output schema
            yield "files", [f.model_dump() for f in files]
            # Always yield conversation_history so user can restore context on fresh sandbox
            yield "conversation_history", conversation_history
            # Always yield session_id so user can continue conversation
            yield "session_id", session_id
            # Always yield sandbox_id (None if disposed) to match Output schema
            yield "sandbox_id", sandbox_id if not input_data.dispose_sandbox else None

        except ClaudeCodeExecutionError as e:
            yield "error", str(e)
            # If sandbox was preserved (dispose_sandbox=False), yield sandbox_id
            # so user can reconnect to or clean up the orphaned sandbox
            if not input_data.dispose_sandbox and e.sandbox_id:
                yield "sandbox_id", e.sandbox_id
        except Exception as e:
            yield "error", str(e)
