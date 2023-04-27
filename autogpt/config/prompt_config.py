# sourcery skip: do-not-use-staticmethod
"""
A module that contains the PromptConfig class object that contains the configuration
"""
import yaml
from autogpt.config.config import Config

CFG = Config()

class PromptConfig:
    """
    A class object that contains the configuration information for the prompt, which will be used by the prompt generator

    Attributes:
        constraints (list): Constraints list for the prompt generator.
        resources (list): Resources list for the prompt generator.
        performance_evaluations (list): Performance evaluation list for the prompt generator.
    """

    def __init__(
        self,
        config_file: str = CFG.prompt_settings_file,
    ) -> None:
        """
        Initialize a class instance with parameters (constraints, resources, performance_evaluations) loaded from
          yaml file if yaml file exists,
        else raises error.

        Parameters:
            constraints (list): Constraints list for the prompt generator.
            resources (list): Resources list for the prompt generator.
            performance_evaluations (list): Performance evaluation list for the prompt generator.
        Returns:
            None
        """
        try:
            with open(config_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise ValueError("Prompt configuration file '" + config_file + "' not found")

        self.constraints = config_params.get("constraints", [])
        self.resources = config_params.get("resources", [])
        self.performance_evaluations = config_params.get("performance_evaluations", [])