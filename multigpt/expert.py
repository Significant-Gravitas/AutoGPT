from __future__ import annotations
import yaml

from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from multigpt.multi_prompt_generator import MultiPromptGenerator
from multigpt.agent_traits import AgentTraits


class Expert(AIConfig):
    expert_instances = []

    def __init__(
            self, ai_name: str = "", ai_role: str = "", ai_goals: list | None = None, ai_traits: AgentTraits = None
    ) -> None:
        super().__init__(ai_name=ai_name, ai_role=ai_role, ai_goals=ai_goals)
        self.ai_traits = ai_traits
        Expert.expert_instances.append(self)

    def __del__(self):
        Expert.expert_instances.remove(self)

    @classmethod
    def get_experts(cls):
        return cls.expert_instances

    @classmethod
    def experts_to_list(cls):
        expert_str = ""
        if cls.expert_instances is None:
            print("Warning. Expert List empty.")
        else:
            for expert in cls.expert_instances:
                expert_str += f"\nName: {expert.ai_name}\n"
                expert_str += f"Role: {expert.ai_role}\n"
        return expert_str

    def to_yaml(self) -> str:
        return yaml.dump(self.__dict__)

    def to_string(self) -> str:
        return f"Name: {self.ai_name}, Role: {self.ai_role}, Goals: {self.ai_goals}"

    def get_prompt_start(self):
        return (
            f"A psychological assessment has produced the following report on your character traits: \n\n {self.ai_traits}"
            "\nAct accordingly in the group discussion.\n\n"
            "Your decisions must always be made independently but you are allowed to collaborate, discuss and disagree"
            " with your team members. Play to your strengths as ChatGPT and pursue"
            " simple strategies with no legal complications."
            f"\n\nYour team consists of: {Expert.experts_to_list()}"
            "\n\nFoster critical discussions but avoid conforming to others' ideas within team collaboration."
        )

    def get_prompt(self):
        """
        This function generates a prompt string that includes various constraints,
            commands, resources, and performance evaluations.

        Returns:
            str: The generated prompt string.
        """

        # Initialize the Config object
        cfg = Config()

        # Initialize the PromptGenerator object
        prompt_generator = MultiPromptGenerator(cfg)

        # Add constraints to the PromptGenerator object
        prompt_generator.add_constraint(
            "~4000 word limit for short term memory. Your short term memory is short, so"
            " immediately save important information to files."
        )
        prompt_generator.add_constraint(
            "If you are unsure how you previously did something or want to recall past"
            " events, thinking about similar events will help you remember."
        )
        prompt_generator.add_constraint("No user assistance")
        prompt_generator.add_constraint(
            'Exclusively use the commands listed in double quotes e.g. "command name"'
        )
        prompt_generator.add_constraint("ALWAYS say something to your team.")

        # Define the command list
        commands = [
            ("Google Search", "google", {"input": "<search>"}),
            (
                "Browse Website",
                "browse_website",
                {"url": "<url>", "question": "<what_you_want_to_find_on_website>"},
            ),
            ("Write to file", "write_to_file", {"file": "<file>", "text": "<text>"}),
            ("Read file", "read_file", {"file": "<file>"}),
            ("Append to file", "append_to_file", {"file": "<file>", "text": "<text>"}),
            ("Delete file", "delete_file", {"file": "<file>"}),
            ("Search Files", "search_files", {"directory": "<directory>"}),
            ("Do Nothing", "do_nothing", {}),
            ("Task Complete (Shutdown)", "task_complete", {"reason": "<reason>"})
        ]

        # Add commands to the PromptGenerator object
        for command_label, command_name, args in commands:
            prompt_generator.add_command(command_label, command_name, args)

        # Add resources to the PromptGenerator object
        prompt_generator.add_resource(
            "Internet access for searches and information gathering."
        )
        prompt_generator.add_resource("Long Term memory management.")
        prompt_generator.add_resource("File output.")

        # Add performance evaluations to the PromptGenerator object
        prompt_generator.add_performance_evaluation(
            "Collaborate with your team but make sure to critically evaluate what they say and"
            " disagree with them if you think they are wrong or if you have a different opinion."
        )
        # prompt_generator.add_performance_evaluation(
        #     "Make sure you and your team are progressing on your common goal."
        # )
        # prompt_generator.add_performance_evaluation(
        #     "If you have the impression one of your team members is getting distracted, help them stay focused."
        # )
        prompt_generator.add_performance_evaluation(
            "If one of your team members is not talking, remind them to participate in the discussion."
        )

        prompt_generator.add_performance_evaluation("You will get one reward token every time you disagree with a team mate.")

        # Generate the prompt string
        return prompt_generator.generate_prompt_string()

    def construct_full_prompt(self) -> str:
        """
        Returns a prompt to the user with the class information in an organized fashion.

        Parameters:
            None

        Returns:
            full_prompt (str): A string containing the initial prompt for the user
              including the ai_name, ai_role and ai_goals.
        """

        # Construct full prompt
        full_prompt = (
            f"You are {self.ai_name}, {self.ai_role}\n{self.get_prompt_start()}\n\nGOALS:\n\n"
        )
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i + 1}. {goal}\n"

        full_prompt += f"\n\n{self.get_prompt()}"
        return full_prompt


    @staticmethod
    def load(config_file: str = "") -> "Expert":
        """
        Returns class object with parameters (ai_name, ai_role, ai_goals) loaded from
          yaml file if yaml file exists,
        else returns class with no parameters.

        Parameters:
           config_file (int): The path to the config yaml file.
             DEFAULT: "../ai_settings.yaml"

        Returns:
            cls (object): An instance of given cls object
        """

        try:
            with open(config_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.SafeLoader)
        except FileNotFoundError:
            config_params = {}

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = config_params.get("ai_goals", [])
        ai_traits = config_params.get("ai_traits", [])
        # type: Type[Expert]
        return Expert(ai_name, ai_role, ai_goals, ai_traits)

    def save(self, config_file: str = "") -> None:
        """
        Saves the class parameters to the specified file yaml file path as a yaml file.

        Parameters:
            config_file(str): The path to the config yaml file.
              DEFAULT: "../ai_settings.yaml"

        Returns:
            None
        """

        config = {
            "ai_name": self.ai_name,
            "ai_role": self.ai_role,
            "ai_goals": self.ai_goals,
            "ai_traits": self.ai_traits
        }
        with open(config_file, "w", encoding="utf-8") as file:
            yaml.dump(config, file, allow_unicode=True)