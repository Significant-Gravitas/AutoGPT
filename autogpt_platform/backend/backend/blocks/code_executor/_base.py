from typing import TYPE_CHECKING, Any, Optional

from e2b_code_interpreter import AsyncSandbox
from e2b_code_interpreter import Result as E2BExecutionResult

from backend.blocks.code_executor.helpers import ProgrammingLanguage
from backend.util.sandbox_files import (
    SandboxFileOutput,
    extract_and_store_sandbox_files,
)

if TYPE_CHECKING:
    from backend.executor.utils import ExecutionContext


class BaseE2BExecutorMixin:
    """Shared implementation methods for E2B executor blocks."""

    # Default working directory in E2B sandboxes
    WORKING_DIR = "/home/user"

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
        execution_context: Optional["ExecutionContext"] = None,
        extract_files: bool = False,
        envs: Optional[dict[str, str]] = None,
    ):
        """
        Create or reconnect to a sandbox, then run setup commands and code.

        Setup commands run on every invocation, including reconnects, so callers
        reusing them must make them idempotent.

        Args:
            extract_files: If True and execution_context provided, extract files
                           created/modified during execution and store to workspace.
        """  # noqa
        sandbox = None
        files: list[SandboxFileOutput] = []
        try:
            if sandbox_id:
                sandbox = await AsyncSandbox.connect(
                    sandbox_id=sandbox_id, api_key=api_key, timeout=timeout
                )
            else:
                sandbox = await AsyncSandbox.create(
                    api_key=api_key, template=template_id, timeout=timeout
                )
            if setup_commands:
                for cmd in setup_commands:
                    await sandbox.commands.run(cmd)

            # Capture timestamp before execution to scope file extraction
            start_timestamp = None
            if extract_files:
                ts_result = await sandbox.commands.run("date -u +%Y-%m-%dT%H:%M:%S")
                start_timestamp = ts_result.stdout.strip() if ts_result.stdout else None

            # Execute the code
            execution = await sandbox.run_code(  # type: ignore[attr-defined]
                code,
                language=language.value,
                envs=envs or {},
                on_error=lambda e: sandbox.kill(),  # Kill the sandbox on error
            )

            if execution.error:
                raise Exception(execution.error)

            results = execution.results
            text_output = execution.text
            stdout_logs = "".join(execution.logs.stdout)
            stderr_logs = "".join(execution.logs.stderr)

            # Extract files created/modified during this execution
            if extract_files and execution_context:
                files = await extract_and_store_sandbox_files(
                    sandbox=sandbox,
                    working_directory=self.WORKING_DIR,
                    execution_context=execution_context,
                    since_timestamp=start_timestamp,
                    text_only=False,  # Include binary files too
                )

            return (
                results,
                text_output,
                stdout_logs,
                stderr_logs,
                sandbox.sandbox_id,
                files,
            )
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
