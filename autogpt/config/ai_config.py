# sourcery skip: do-not-use-staticmethod
"""
A module that contains the AIConfig class object that contains the configuration
"""
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional, Type

import distro
import yaml

from autogpt.prompts.generator import PromptGenerator

# Soon this will go in a folder where it remembers more stuff about the run(s)
SAVE_FILE = str(Path(os.getcwd()) / "ai_settings.yaml")
MAX_AI_CONFIG = 5

from .singleton import AbstractSingleton

class AIConfig(AbstractSingleton):
    """
    A class object that contains the configuration information for the AI

    Attributes:
        __current_config (int): The number of the current config from 0 to 4.
        __configs (list): A list of Dictionary (configs) :
            {"ai_name": ai_name, 
            "ai_role": ai_role, 
            "ai_goals": ai_goals, 
            "prompt_generator": prompt_generator,
            "command_registry": command_registry 
            })
        # TODO __version (str): Version of the app
    """

        
    def __init__(self, config_number : int = -1,config_file: str = SAVE_FILE) -> None:
        """
        Initialize a class instance
        Parameters:
            config_number(int): The config entry number (0 to 4) to be updated or added.
            config_file(str): The path to the config yaml file.
              DEFAULT: "../ai_settings.yaml"
        Returns:
            None
        """
        self.config_file = config_file
        self.__current_config = config_number
        self.load(self.config_file)
        if (config_number != -1) : 
            self.set_config_number(config_number)


    @classmethod
    def load(cls, config_file: str = SAVE_FILE) -> list:
        """
        Loads the configurations from the specified YAML file and returns a list of dictionaries containing the configuration parameters.

        Parameters:
            config_file(str): The path to the config yaml file.
            DEFAULT: "../ai_settings.yaml"
        
        Returns:
            ai_configs (list): A list of dictionaries containing the configuration parameters for each entry.
        """
        try:
            with open(config_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config_params = {}

        cls.__configs = []
        for key, row in config_params.items():
            if key == 'version' :
                continue
            elif key == 'configurations' : 
                for yaml_config in row  :
                    print(yaml_config)
                    config = {'ai_name': yaml_config["ai_name"],
                            "ai_role": yaml_config["ai_role"],
                            "ai_goals": yaml_config["ai_goals"],
                            "prompt_generator": yaml_config["prompt_generator"],
                            "command_registry": yaml_config["command_registry"]}
                    cls.__configs.append(config)

                # ai_name = row.get("ai_name", "")
                # ai_role = row.get("ai_role", "")
                # ai_goals = row.get("ai_goals", [])
                # prompt_generator = row.get("prompt_generator", None)
                # command_registry = row.get("command_registry", None)
                # cls.__configs.append({"ai_name": ai_name, "ai_role": ai_role, "ai_goals": ai_goals, "prompt_generator": prompt_generator,"command_registry": command_registry })
        return cls.__configs

    def save(self, config_file: str = SAVE_FILE) -> None:
            """
            Saves the class parameters to the specified file yaml file path as a yaml file.

            Parameters:
                config_number(int): The config entry number (1 to 5) to be updated or added.
                config_file(str): The path to the config yaml file.
                DEFAULT: "../ai_settings.yaml"
            Returns:
                None
            """

            with open(config_file, "w", encoding="utf-8") as file:
                yaml.dump(self.__configs, file, allow_unicode=True)

    def construct_full_prompt(
        self, prompt_generator: Optional[PromptGenerator] = None
    ) -> str:
        """
        Returns a prompt to the user with the class information in an organized fashion.

        Parameters:
            None

        Returns:
            full_prompt (str): A string containing the initial prompt for the user
                including the ai_name, ai_role and ai_goals.
        """

        prompt_start = (
            "Your decisions must always be made independently without"
            " seeking user assistance. Play to your strengths as an LLM and pursue"
            " simple strategies with no legal complications."
            ""
        )

        from autogpt.config import Config
        from autogpt.prompts.prompt import build_default_prompt_generator

        cfg = Config()
        if prompt_generator is None:
            prompt_generator = build_default_prompt_generator()
        prompt_generator.goals = self.__configs[self.__current_config]["ai_goals"]
        prompt_generator.name = self.__configs[self.__current_config]["ai_name"]
        prompt_generator.role = self.__configs[self.__current_config]["ai_role"]
        prompt_generator.command_registry = self.__configs[self.__current_config]["command_registry"]
        for plugin in cfg.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            prompt_generator = plugin.post_prompt(prompt_generator)

        if cfg.execute_local_commands:
            # add OS info to prompt
            os_name = platform.system()
            os_info = (
                platform.platform(terse=True)
                if os_name != "Linux"
                else distro.name(pretty=True)
            )

            prompt_start += f"\nThe OS you are running on is: {os_info}"

        # Construct full prompt
        full_prompt = f"You are {prompt_generator.name}, {prompt_generator.role}\n{prompt_start}\n\nGOALS:\n\n"
        for i, goal in enumerate(self.__configs[self.__current_config]["ai_goals"]):
            full_prompt += f"{i+1}. {goal}\n"
        self.__configs[self.__current_config]["prompt_generator"] = prompt_generator
        full_prompt += f"\n\n{prompt_generator.generate_prompt_string()}"
        return full_prompt
        
    def set_config_number(self, new_config_number: int) -> None:
        """
        Sets the current configuration number.

        Parameters:
            new_config_number (int): The new configuration number to be set.

        Returns:
            None
        """
        if new_config_number < 0 or new_config_number >= len(self.__configs):
            raise ValueError(f"set_config_number: Value must be between 0 and {len(self.__configs)-1}")
        self.__current_config = new_config_number
        return

    def get_config_number(self) -> int:
        """
        Gets the current configuration number.

        Parameters:
            None

        Returns:
            current_config (int): The current configuration number.
        """
        return self.__current_config

    def get_current_config(self) :
        """
        Gets the current configuration dictionary.

        Parameters:
            None

        Returns:
            current_config (dict): The current configuration dictionary.
        """
        if (self.__current_config == -1) :
            raise ValueError(f"get_current_config: Value {self.__current_config } not expected")
        return self.__configs[self.__current_config]

    def set_config(self, 
                    config_number : int, 
                    ai_name : str, 
                    ai_role : str, 
                    ai_goals : list, 
                    prompt_generator : str, 
                    command_registry : str ,
                    overwrite : bool = False) -> bool:
        """
        Sets the configuration parameters for the given configuration number.

        Parameters:
            config_number (int): The configuration number to be updated.
            ai_name (str): The AI name.
            ai_role (str): The AI role.
            ai_goals (list): The AI goals.
            prompt_generator (Optional[PromptGenerator]): The prompt generator instance.
            command_registry: The command registry instance.

        Returns:
            success (bool): True if the configuration is successfully updated, False otherwise.
        """
        number_of_config = len(self.__configs) 
        if config_number == None  and number_of_config < MAX_AI_CONFIG: 
            config_number = number_of_config
        
        if config_number > MAX_AI_CONFIG :
            raise ValueError(f"set_config: Value {config_number} not expected")
        

        # Check for duplicate config names
        for idx, config in enumerate(self.__configs):
            if config["ai_name"] == ai_name and idx != config_number:
                print(f"Config with the name '{ai_name}' already exists")
                return False
        


                
        if (config_number >= number_of_config ) :
            self.__configs.append({"ai_name": ai_name, "ai_role": ai_role, "ai_goals": ai_goals, "prompt_generator": prompt_generator,"command_registry": command_registry })
        else :
            self.__configs[config_number]["ai_name"] = ai_name,
            self.__configs[config_number]["ai_role"] = ai_role
            self.__configs[config_number]["ai_goals"] = ai_goals

            self.__configs[config_number]["prompt_generator"]= prompt_generator
            self.__configs[config_number]["command_registry"]= command_registry

        self.set_config_number(new_config_number = config_number)

        return True

    def get_configs(self):
        """
        Gets the list of all configuration dictionaries.

        Parameters:
            None

        Returns:
            configs (list): A list of all configuration dictionaries.
        """

        return self.__configs