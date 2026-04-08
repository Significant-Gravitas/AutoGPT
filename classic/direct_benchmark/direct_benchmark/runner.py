"""AgentRunner - manages single agent lifecycle for a challenge."""

import asyncio
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, cast

if TYPE_CHECKING:
    from forge.llm.providers import ModelName

from autogpt.agent_factory.configurators import create_agent
from autogpt.agents.agent import Agent
from autogpt.app.config import ConfigBuilder

from forge.file_storage import FileStorageBackendName, get_storage
from forge.llm.providers import MultiProvider

from .models import BenchmarkConfig, Challenge, ChallengeResult, StepResult

# Type for step logging callback
StepCallback = Callable[[str, str, int, str, str, bool], None]
# Args: config_name, challenge_name, step_num, tool_name, result_preview, is_error


class AgentRunner:
    """Runs a single agent instance for a challenge."""

    def __init__(
        self,
        config: BenchmarkConfig,
        workspace_root: Path,
        no_cutoff: bool = False,
        step_callback: Optional[StepCallback] = None,
    ):
        self.config = config
        self.workspace_root = workspace_root
        self.no_cutoff = no_cutoff
        self.step_callback = step_callback
        self._agent: Optional[Agent] = None
        self._workspace: Optional[Path] = None
        self._llm_provider: Optional[MultiProvider] = None

    async def run_challenge(
        self, challenge: Challenge, attempt: int = 1
    ) -> ChallengeResult:
        """Run a single challenge and return the result."""
        start_time = datetime.now()
        steps: list[StepResult] = []

        # Create isolated workspace for this run
        prefix = f"{challenge.name}_{self.config.strategy}_"
        if attempt > 1:
            prefix = f"{challenge.name}_{self.config.strategy}_attempt{attempt}_"

        self._workspace = Path(
            tempfile.mkdtemp(
                prefix=prefix,
                dir=self.workspace_root,
            )
        )

        try:
            # Copy input artifacts to workspace
            self._setup_workspace(challenge)

            # Create the agent
            agent = await self._create_agent(challenge)

            # Determine timeout
            if self.no_cutoff:
                timeout = None  # No timeout
            else:
                timeout = min(challenge.cutoff, self.config.timeout_seconds)

            # Run the agent loop
            result = await self._run_agent_loop(
                agent,
                challenge,
                steps,
                timeout=timeout,
                attempt=attempt,
            )

            return result

        except asyncio.TimeoutError:
            # Get cost even on timeout
            cost = 0.0
            if self._llm_provider:
                cost = self._llm_provider.get_incurred_cost()
            return ChallengeResult(
                challenge_name=challenge.name,
                config_name=self.config.config_name,
                attempt=attempt,
                success=False,
                score=0.0,
                steps=steps,
                n_steps=len(steps),
                run_time_seconds=(datetime.now() - start_time).total_seconds(),
                cost=cost,
                timed_out=True,
                error_message="Challenge timed out",
            )
        except Exception as e:
            import traceback

            # Get cost even on error
            cost = 0.0
            if self._llm_provider:
                cost = self._llm_provider.get_incurred_cost()
            return ChallengeResult(
                challenge_name=challenge.name,
                config_name=self.config.config_name,
                attempt=attempt,
                success=False,
                score=0.0,
                steps=steps,
                n_steps=len(steps),
                run_time_seconds=(datetime.now() - start_time).total_seconds(),
                cost=cost,
                error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )
        finally:
            await self._cleanup()

    def _get_agent_id(self, challenge: Challenge) -> str:
        """Get the deterministic agent ID for a challenge."""
        return f"benchmark-{challenge.name}-{self.config.strategy}"

    def _setup_workspace(self, challenge: Challenge) -> None:
        """Set up workspace with input artifacts."""
        if self._workspace is None:
            return

        # The agent stores files in {workspace}/.autogpt/agents/{agent_id}/workspace/
        # We need to pre-create this directory and copy artifacts there
        agent_id = self._get_agent_id(challenge)
        agent_workspace = (
            self._workspace / ".autogpt" / "agents" / agent_id / "workspace"
        )
        agent_workspace.mkdir(parents=True, exist_ok=True)

        # Copy artifacts_in if it exists
        artifacts_in = challenge.artifacts_dir / "artifacts_in"
        if artifacts_in.exists():
            for item in artifacts_in.iterdir():
                dest = agent_workspace / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest)

    async def _create_agent(self, challenge: Challenge) -> Agent:
        """Create and configure the agent."""
        if self._workspace is None:
            raise RuntimeError("Workspace not initialized")

        # Build AppConfig from environment + model config overrides
        app_config = ConfigBuilder.build_config_from_env(workspace=self._workspace)

        # Apply model and strategy configuration
        if self.config.model.smart_llm:
            app_config.smart_llm = cast("ModelName", self.config.model.smart_llm)
        if self.config.model.fast_llm:
            app_config.fast_llm = cast("ModelName", self.config.model.fast_llm)
        app_config.prompt_strategy = self.config.strategy
        app_config.noninteractive_mode = True
        app_config.continuous_mode = True

        if self.config.model.thinking_budget_tokens:
            app_config.thinking_budget_tokens = self.config.model.thinking_budget_tokens
        if self.config.model.reasoning_effort:
            app_config.reasoning_effort = self.config.model.reasoning_effort

        # Set up file storage
        data_dir = self._workspace / ".autogpt"
        file_storage = get_storage(
            FileStorageBackendName.LOCAL,
            root_path=data_dir,
            restrict_to_root=True,
        )
        file_storage.initialize()

        # Create LLM provider
        llm_provider = MultiProvider()

        # Create agent
        agent_id = self._get_agent_id(challenge)
        agent = create_agent(
            agent_id=agent_id,
            task=challenge.task,
            app_config=app_config,
            file_storage=file_storage,
            llm_provider=llm_provider,
        )

        # Enable local command execution for benchmarks
        # Use denylist mode to block dangerous commands while allowing flexibility
        if hasattr(agent, "code_executor"):
            agent.code_executor.config.execute_local_commands = True
            agent.code_executor.config.shell_command_control = "denylist"
            agent.code_executor.config.shell_denylist = [
                "rm",  # Block file removal
                "sudo",  # Block privilege escalation
                "chmod",  # Block permission changes
                "chown",  # Block ownership changes
                "mkfs",  # Block filesystem creation
                "dd",  # Block disk operations
                "kill",  # Block process killing
                "pkill",  # Block process killing
                "killall",  # Block process killing
                "reboot",  # Block system reboot
                "shutdown",  # Block system shutdown
                "poweroff",  # Block system poweroff
                "halt",  # Block system halt
                "init",  # Block init commands
                "systemctl",  # Block systemd commands
                "service",  # Block service commands
            ]

        # Disable clipboard commands for benchmarks - they add overhead without value
        app_config.disabled_commands = [
            "clipboard_copy",
            "clipboard_paste",
            "clipboard_list",
            "clipboard_clear",
        ]

        self._agent = agent
        self._llm_provider = llm_provider
        return agent

    async def _run_agent_loop(
        self,
        agent: Agent,
        challenge: Challenge,
        steps: list[StepResult],
        timeout: Optional[int],
        attempt: int = 1,
    ) -> ChallengeResult:
        """Run the agent loop until completion or timeout."""
        start_time = datetime.now()
        cumulative_cost = 0.0

        async def run_loop() -> bool:
            """Run the agent loop. Returns True if finished normally."""
            nonlocal cumulative_cost

            for step_num in range(self.config.max_steps):
                # Propose next action
                proposal = await agent.propose_action()

                # Get cumulative cost from LLM provider
                if self._llm_provider:
                    cumulative_cost = self._llm_provider.get_incurred_cost()

                # Check for finish command - record it and return
                if proposal.use_tool.name == "finish":
                    steps.append(
                        StepResult(
                            step_num=step_num + 1,
                            tool_name=proposal.use_tool.name,
                            tool_args=proposal.use_tool.arguments,
                            result="Agent finished",
                            is_error=False,
                            cumulative_cost=cumulative_cost,
                        )
                    )
                    return True

                # Execute the action
                result = await agent.execute(proposal)

                # Update cost after execution
                if self._llm_provider:
                    cumulative_cost = self._llm_provider.get_incurred_cost()

                # Get result info
                result_str = str(getattr(result, "outputs", result))
                is_error = hasattr(result, "status") and result.status == "error"

                # Record step
                steps.append(
                    StepResult(
                        step_num=step_num + 1,
                        tool_name=proposal.use_tool.name,
                        tool_args=proposal.use_tool.arguments,
                        result=result_str,
                        is_error=is_error,
                        cumulative_cost=cumulative_cost,
                    )
                )

                # Call step callback if provided
                if self.step_callback:
                    # Truncate result for display
                    result_preview = (
                        result_str[:100] + "..."
                        if len(result_str) > 100
                        else result_str
                    )
                    result_preview = result_preview.replace("\n", " ")
                    self.step_callback(
                        self.config.config_name,
                        challenge.name,
                        step_num + 1,
                        proposal.use_tool.name,
                        result_preview,
                        is_error,
                    )

            return False  # Hit max steps

        # Run with or without timeout
        if timeout is not None and timeout > 0:
            finished = await asyncio.wait_for(run_loop(), timeout=timeout)
        else:
            finished = await run_loop()

        run_time = (datetime.now() - start_time).total_seconds()

        # Collect output files for evaluation
        output_files = self._collect_output_files()

        return ChallengeResult(
            challenge_name=challenge.name,
            config_name=self.config.config_name,
            attempt=attempt,
            success=False,  # Will be set by evaluator
            score=0.0,  # Will be set by evaluator
            steps=steps,
            n_steps=len(steps),
            run_time_seconds=run_time,
            cost=cumulative_cost,
            timed_out=not finished and len(steps) >= self.config.max_steps,
            output_files=output_files,
        )

    def _collect_output_files(self) -> dict[str, str]:
        """Collect output files from workspace for evaluation."""
        outputs: dict[str, str] = {}

        if self._workspace is None:
            return outputs

        # Check agent workspace directory
        agent_workspace = self._workspace / ".autogpt" / "agents"

        # Find agent workspace directory
        if agent_workspace.exists():
            for agent_dir in agent_workspace.iterdir():
                workspace_dir = agent_dir / "workspace"
                if workspace_dir.exists():
                    for file in workspace_dir.rglob("*"):
                        if file.is_file():
                            try:
                                rel_path = file.relative_to(workspace_dir)
                                content = file.read_text(errors="replace")
                                outputs[str(rel_path)] = content
                            except Exception:
                                pass

        # Also check the root workspace for any files created there
        for file in self._workspace.iterdir():
            if file.is_file() and not file.name.startswith("."):
                try:
                    outputs[file.name] = file.read_text(errors="replace")
                except Exception:
                    pass

        return outputs

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._agent = None
        # Note: We don't delete workspace here to preserve results for debugging
        # The harness can clean up workspaces after collecting results
