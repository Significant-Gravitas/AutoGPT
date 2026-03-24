"""A factory for constructing agents with predefined configurations."""
from __future__ import annotations

from autogpt.agent.agent import Agent
from autogpt.agent.build_agent import BuildAgent
from autogpt.config import Config
from autogpt.memory import get_memory


AGENT_TEMPLATES = {
    "build": {
        "ai_name": "BuildGPT",
        "ai_role": "an AI developer agent that builds, tests, and improves software projects",
        "ai_goals": [
            "Analyze the project structure and understand the codebase",
            "Identify and fix bugs or issues in the code",
            "Write and run tests to ensure code quality",
            "Improve code based on best practices and conventions",
            "Complete the assigned build task efficiently",
        ],
    },
    "research": {
        "ai_name": "ResearchGPT",
        "ai_role": "an AI research agent that gathers information and produces reports",
        "ai_goals": [
            "Search for relevant information on the given topic",
            "Analyze and synthesize findings from multiple sources",
            "Produce a clear and well-structured summary",
            "Identify key insights and actionable recommendations",
            "Save the research report to a file",
        ],
    },
    "code-review": {
        "ai_name": "ReviewGPT",
        "ai_role": "an AI code review agent that analyzes code for quality and correctness",
        "ai_goals": [
            "Read and understand the codebase thoroughly",
            "Identify bugs, security issues, and code smells",
            "Suggest improvements following best practices",
            "Evaluate test coverage and suggest missing tests",
            "Produce a detailed code review report",
        ],
    },
}

DEFAULT_TRIGGERING_PROMPT = (
    "Determine which next command to use, and respond using the"
    " format specified above:"
)


class AgentBuilder:
    """Factory for constructing configured Agent instances.

    Provides a fluent builder interface and predefined templates for
    creating specialized agents.

    Example:
        agent = (
            AgentBuilder()
            .with_name("MyBuilder")
            .with_role("a software build agent")
            .with_goals(["Build the project", "Run tests"])
            .with_project_dir("/path/to/project")
            .build_agent(agent_type="build")
        )
        agent.start_interaction_loop()
    """

    def __init__(self):
        self._ai_name = ""
        self._ai_role = ""
        self._ai_goals = []
        self._project_dir = ""
        self._build_config = {}
        self._memory_type = None
        self._next_action_count = 0
        self._continuous_mode = False
        self._continuous_limit = 0

    def with_name(self, name: str) -> AgentBuilder:
        """Set the agent name."""
        self._ai_name = name
        return self

    def with_role(self, role: str) -> AgentBuilder:
        """Set the agent role description."""
        self._ai_role = role
        return self

    def with_goals(self, goals: list[str]) -> AgentBuilder:
        """Set the agent goals."""
        self._ai_goals = goals
        return self

    def with_project_dir(self, project_dir: str) -> AgentBuilder:
        """Set the project directory for build agents."""
        self._project_dir = project_dir
        return self

    def with_build_config(self, config: dict) -> AgentBuilder:
        """Set build-specific configuration."""
        self._build_config = config
        return self

    def with_memory_type(self, memory_type: str) -> AgentBuilder:
        """Set the memory backend type."""
        self._memory_type = memory_type
        return self

    def with_next_action_count(self, count: int) -> AgentBuilder:
        """Set the number of continuous actions to execute."""
        self._next_action_count = count
        return self

    def from_template(self, template_name: str) -> AgentBuilder:
        """Load settings from a predefined template.

        Args:
            template_name: One of 'build', 'research', or 'code-review'.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If template_name is not recognized.
        """
        if template_name not in AGENT_TEMPLATES:
            available = ", ".join(AGENT_TEMPLATES.keys())
            raise ValueError(
                f"Unknown template '{template_name}'. Available: {available}"
            )

        template = AGENT_TEMPLATES[template_name]
        self._ai_name = template["ai_name"]
        self._ai_role = template["ai_role"]
        self._ai_goals = list(template["ai_goals"])
        return self

    def build_agent(self, agent_type: str = "default") -> Agent:
        """Construct and return the configured agent.

        Args:
            agent_type: The type of agent to build.
                'build' creates a BuildAgent with development-focused prompts.
                'default' creates a standard Agent.

        Returns:
            A configured Agent or BuildAgent instance.
        """
        cfg = Config()
        memory = get_memory(cfg, init=True)

        if agent_type == "build":
            system_prompt = BuildAgent.get_build_prompt(
                ai_name=self._ai_name,
                ai_role=self._ai_role,
                ai_goals=self._ai_goals,
                project_dir=self._project_dir,
            )
            return BuildAgent(
                ai_name=self._ai_name,
                memory=memory,
                full_message_history=[],
                next_action_count=self._next_action_count,
                system_prompt=system_prompt,
                triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
                project_dir=self._project_dir,
                build_config=self._build_config,
            )

        # Default agent - use standard prompt construction
        from autogpt.config.ai_config import AIConfig

        ai_config = AIConfig(
            ai_name=self._ai_name,
            ai_role=self._ai_role,
            ai_goals=self._ai_goals,
        )
        system_prompt = ai_config.construct_full_prompt()

        return Agent(
            ai_name=self._ai_name,
            memory=memory,
            full_message_history=[],
            next_action_count=self._next_action_count,
            system_prompt=system_prompt,
            triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        )

    @staticmethod
    def list_templates() -> list[str]:
        """Return available template names."""
        return list(AGENT_TEMPLATES.keys())
