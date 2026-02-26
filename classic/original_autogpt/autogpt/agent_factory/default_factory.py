"""Default implementation of AgentFactory for sub-agent spawning.

This factory creates Agent instances for use as sub-agents within
a prompt strategy. It follows the same pattern as the direct_benchmark
runner for agent creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings

from forge.agent.execution_context import AgentFactory, ExecutionContext
from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile

if TYPE_CHECKING:
    from autogpt.app.config import AppConfig, PromptStrategyName


class DefaultAgentFactory(AgentFactory):
    """Default implementation of AgentFactory.

    Creates Agent instances for sub-agent spawning. Reuses the pattern
    from direct_benchmark/runner.py for agent creation.

    The factory is stateless - all configuration comes from the AppConfig
    and ExecutionContext.
    """

    def __init__(self, app_config: "AppConfig"):
        """Initialize the factory.

        Args:
            app_config: The application configuration to use for
                creating agents. This provides LLM settings, disabled
                commands, etc.
        """
        self.app_config = app_config

    def create_agent(
        self,
        agent_id: str,
        task: str,
        context: ExecutionContext,
        ai_profile: Optional[AIProfile] = None,
        directives: Optional[AIDirectives] = None,
        strategy: Optional[str] = None,
    ) -> Agent:
        """Create a new agent instance for sub-agent execution.

        Args:
            agent_id: Unique identifier for the agent.
            task: The task the agent should accomplish.
            context: Execution context with shared resources.
            ai_profile: Optional AI profile override. If not provided,
                a default profile is created.
            directives: Optional directives override. If not provided,
                default directives are used.
            strategy: Optional strategy name override (e.g., "one_shot").
                If not provided, uses the app_config default.

        Returns:
            A new Agent instance configured for the task.
        """
        # Create default profile if not provided
        if ai_profile is None:
            ai_profile = AIProfile(
                ai_name=f"SubAgent-{agent_id[:8]}",
                ai_role="A specialized sub-agent working on a specific task.",
            )

        # Create default directives if not provided
        if directives is None:
            directives = AIDirectives(
                constraints=[
                    "Focus only on the assigned task.",
                    "Do not ask for user input - work autonomously.",
                    "Complete the task efficiently and call finish when done.",
                ],
                resources=[
                    "The same tools as your parent agent.",
                ],
                best_practices=[
                    "Think step by step.",
                    "Be concise in your outputs.",
                ],
            )

        # Create agent settings
        agent_state = self._create_agent_state(
            agent_id=agent_id,
            task=task,
            ai_profile=ai_profile,
            directives=directives,
        )

        # Copy app config and optionally override strategy
        config = self.app_config.model_copy(deep=True)
        if strategy:
            config.prompt_strategy = cast("PromptStrategyName", strategy)

        # Sub-agents should always be non-interactive
        config.noninteractive_mode = True
        config.continuous_mode = True

        # Create the agent with the provided execution context
        return Agent(
            settings=agent_state,
            llm_provider=context.llm_provider,
            file_storage=context.file_storage,
            app_config=config,
            execution_context=context,
        )

    def _create_agent_state(
        self,
        agent_id: str,
        task: str,
        ai_profile: AIProfile,
        directives: AIDirectives,
    ) -> AgentSettings:
        """Create the agent settings/state object.

        Args:
            agent_id: Unique identifier for the agent.
            task: The task the agent should accomplish.
            ai_profile: The AI profile for this agent.
            directives: The directives for this agent.

        Returns:
            AgentSettings configured for the sub-agent.
        """
        return AgentSettings(
            agent_id=agent_id,
            name=Agent.default_settings.name,
            description=Agent.default_settings.description,
            task=task,
            ai_profile=ai_profile,
            directives=directives,
            config=AgentConfiguration(
                fast_llm=self.app_config.fast_llm,
                smart_llm=self.app_config.smart_llm,
                allow_fs_access=not self.app_config.restrict_to_workspace,
            ),
            history=Agent.default_settings.history.model_copy(deep=True),
        )
