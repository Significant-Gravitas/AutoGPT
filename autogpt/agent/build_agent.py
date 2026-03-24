"""A specialized agent for software development and build tasks."""
from colorama import Fore

from autogpt.agent.agent import Agent
from autogpt.config import Config
from autogpt.logs import logger
from autogpt.promptgenerator import PromptGenerator


class BuildAgent(Agent):
    """An agent specialized for software development and build tasks.

    Extends the base Agent with build-specific system prompts, constraints,
    and a focused command set for code evaluation, testing, and file operations.

    Attributes:
        project_dir: The root directory of the project to work on.
        build_config: Optional dict of build-specific configuration overrides.
    """

    def __init__(
        self,
        ai_name,
        memory,
        full_message_history,
        next_action_count,
        system_prompt,
        triggering_prompt,
        project_dir="",
        build_config=None,
    ):
        super().__init__(
            ai_name=ai_name,
            memory=memory,
            full_message_history=full_message_history,
            next_action_count=next_action_count,
            system_prompt=system_prompt,
            triggering_prompt=triggering_prompt,
        )
        self.project_dir = project_dir
        self.build_config = build_config or {}

    @staticmethod
    def get_build_prompt(ai_name, ai_role, ai_goals, project_dir=""):
        """Generate a build-focused system prompt.

        Args:
            ai_name: Name of the build agent.
            ai_role: Role description for the agent.
            ai_goals: List of build objectives.
            project_dir: Root directory of the project.

        Returns:
            str: The complete system prompt for the build agent.
        """
        cfg = Config()
        prompt_generator = PromptGenerator()

        # Build-specific constraints
        prompt_generator.add_constraint(
            "~4000 word limit for short term memory. Save important information to files."
        )
        prompt_generator.add_constraint(
            "Focus on code quality, testing, and reproducible builds."
        )
        prompt_generator.add_constraint(
            "Always validate code changes by running tests before considering a task complete."
        )
        prompt_generator.add_constraint(
            "Use version control best practices when modifying code."
        )
        prompt_generator.add_constraint(
            'Exclusively use the commands listed in double quotes e.g. "command name"'
        )

        # Build-focused commands
        commands = [
            ("Evaluate Code", "evaluate_code", {"code": "<full_code_string>"}),
            (
                "Get Improved Code",
                "improve_code",
                {"suggestions": "<list_of_suggestions>", "code": "<full_code_string>"},
            ),
            (
                "Write Tests",
                "write_tests",
                {"code": "<full_code_string>", "focus": "<list_of_focus_areas>"},
            ),
            ("Execute Python File", "execute_python_file", {"file": "<file>"}),
            ("Write to file", "write_to_file", {"file": "<file>", "text": "<text>"}),
            ("Read file", "read_file", {"file": "<file>"}),
            ("Append to file", "append_to_file", {"file": "<file>", "text": "<text>"}),
            ("Delete file", "delete_file", {"file": "<file>"}),
            ("Search Files", "search_files", {"directory": "<directory>"}),
            ("Google Search", "google", {"input": "<search>"}),
            (
                "Browse Website",
                "browse_website",
                {"url": "<url>", "question": "<what_you_want_to_find_on_website>"},
            ),
            (
                "Clone Repository",
                "clone_repository",
                {"repository_url": "<url>", "clone_path": "<directory>"},
            ),
            ("Do Nothing", "do_nothing", {}),
            ("Task Complete (Shutdown)", "task_complete", {"reason": "<reason>"}),
        ]

        if cfg.execute_local_commands:
            commands.insert(
                -2,
                (
                    "Execute Shell Command, non-interactive commands only",
                    "execute_shell",
                    {"command_line": "<command_line>"},
                ),
            )

        for command_label, command_name, args in commands:
            prompt_generator.add_command(command_label, command_name, args)

        # Build-specific resources
        prompt_generator.add_resource(
            "Internet access for searching documentation and solutions."
        )
        prompt_generator.add_resource("Long Term memory management.")
        prompt_generator.add_resource("File system access for reading and writing code.")
        prompt_generator.add_resource(
            "Code evaluation and improvement capabilities."
        )

        # Build-specific performance evaluations
        prompt_generator.add_performance_evaluation(
            "Ensure all code changes pass existing tests before proceeding."
        )
        prompt_generator.add_performance_evaluation(
            "Write clean, maintainable code that follows project conventions."
        )
        prompt_generator.add_performance_evaluation(
            "Minimize the number of file operations by planning changes carefully."
        )
        prompt_generator.add_performance_evaluation(
            "Every command has a cost, so be smart and efficient."
        )

        prompt_start = (
            "Your decisions must always be made independently without"
            " seeking user assistance. You are a software development agent"
            " focused on building, testing, and improving code."
        )

        if project_dir:
            prompt_start += f" Your project directory is: {project_dir}"

        full_prompt = (
            f"You are {ai_name}, {ai_role}\n{prompt_start}\n\nGOALS:\n\n"
        )
        for i, goal in enumerate(ai_goals):
            full_prompt += f"{i + 1}. {goal}\n"

        full_prompt += f"\n\n{prompt_generator.generate_prompt_string()}"
        return full_prompt

    def start_interaction_loop(self):
        """Start the build agent interaction loop with build-specific logging."""
        logger.typewriter_log(
            "BUILD AGENT STARTED: ",
            Fore.GREEN,
            f"{self.ai_name}",
        )
        if self.project_dir:
            logger.typewriter_log(
                "Project Directory: ",
                Fore.CYAN,
                self.project_dir,
            )
        super().start_interaction_loop()
