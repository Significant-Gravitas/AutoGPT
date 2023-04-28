"""
This module provides a class for configuring an AI agent's settings.

Description:
- This module contains the `AgentConfig` class, which represents the configuration settings for an AI agent.


Functions: None

Global Variables: None

Classes:
- AgentConfig: A class representing the configuration settings for an AI agent.

Dependencies: 
- typing: A built-in module for type hints and annotations.


"""


import os
import yaml
from pathlib import Path
from typing import Optional, Type, List
try :
    from autogpt.projects.agent_model import AgentModel
    from autogpt.projects.projects_broker import ProjectsBroker
except ImportError as e:
    pass

# Soon this will go in a folder where it remembers more stuff about the run(s)
# @TODO SAVE_FILE = str(Path(os.getcwd()) / "ai_settings.yaml")

class AgentModel(): 
    """
    A class representing the configuration settings for an AI agent.

    Attributes:
        agent_name (str): The name of the agent.
        agent_goals (List[str]): A list of the agent's goals.
        agent_role (str): The role of the agent.
        agent_model (Optional[str]): The name of the agent's model, if applicable.
        agent_model_type (Optional[str]): The type of the agent's model, if applicable.
        team_name (Optional[str]): The name of the team that the agent belongs to, if applicable.
        prompt_generator (Optional[Any]): An instance of the `PromptGenerator` class used to generate prompts for the user.
        command_registry (Optional[Any]): An instance of the `CommandRegistry` class used to manage the available commands for the agent.

    Methods:
        __init__(self, agent_name: str, agent_goals: List[str], agent_role: str, agent_model: Optional[str] = None,
        agent_model_type: Optional[str] = None, team_name: Optional[str] = None,
        prompt_generator: Optional[Any] = None, command_registry: Optional[Any] = None) -> None
            Initializes the `AgentModel` instance with the given attributes.

        load_agent(cls, agent_data: dict) -> "AgentModel":
            Loads agent data from a dictionary and returns an `AgentModel` instance.

        save(self) -> dict:
            Saves the `AgentModel` object as a dictionary representation.

        load(config_file: str = '') -> "AgentModel":  
            DEPRECATED - THIS IS TO MAINTAIN BACKWARD COMPATIBILITY
            Returns class object with parameters (ai_name, ai_role, ai_goals, api_budget) loaded from
            yaml file if yaml file exists, else returns class with no parameters.
            Parameters:
               config_file (int): The path to the config yaml file.
            Returns:
                cls (object): An instance of the `AgentModel` class.
    """
        
    def __init__(self, 
                 agent_name: str, 
                 agent_goals: List, 
                 agent_role: str, 
                 agent_model: str, 
                 agent_model_type: str,
                 team_name: Optional[str] = None,
                 prompt_generator =  None,
                command_registry =  None
                ) -> None:
        """
        Initializes the AgentModel instance with the given attributes.

        Args:
            agent_name (str): The name of the agent.
            agent_goals (List[str]): A list of the agent's goals.
            agent_role (str): The role of the agent.
            agent_model (Optional[str], optional): The name of the agent's model, if applicable. Defaults to None.
            agent_model_type (Optional[str], optional): The type of the agent's model, if applicable. Defaults to None.
            team_name (Optional[str], optional): The name of the team that the agent belongs to, if applicable. Defaults to None.
            prompt_generator (Optional[Any], optional): An instance of the `PromptGenerator` class used to generate prompts for the user. Defaults to None.
            command_registry (Optional[Any], optional): An instance of the `CommandRegistry` class used to manage the available commands for the agent. Defaults to None.
        """
        self.agent_name = agent_name
        self.agent_goals = agent_goals
        self.agent_role = agent_role
        self.agent_model = agent_model
        self.agent_model_type = agent_model_type
        self.team_name = team_name
        self.prompt_generator= prompt_generator
        self.command_registry= command_registry

    @classmethod
    def load_agent(cls, agent_data: dict) -> "AgentModel":
        """
        Loads agent data from a dictionary and returns an `AgentModel` instance.

        Args:
            agent_data (dict): A dictionary containing the agent's data.

        Returns:
            agent_instance (AgentModel): An `AgentModel` instance with the loaded data.
        """
        agent_name = agent_data["agent_name"]
        agent_goals = [goal["goal_name"] for goal in agent_data["agent_goals"]]
        agent_role = agent_data["agent_role"]
        agent_model = agent_data["agent_model"]
        agent_model_type = agent_data["agent_model_type"]
        team_name = agent_data.get("team_name")

        return  cls(
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            agent_model=agent_model,
            agent_model_type=agent_model_type,
            team_name=team_name)
    
    def save(self) -> dict:
        """
        Saves the `AgentModel` object as a dictionary representation.

        Returns:
            agent_dict (dict): A dictionary representation of the `AgentModel` object.
        """
        agent_dict = {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_goals": self.agent_goals,
            "agent_model": self.agent_model,
            "agent_model_type": self.agent_model_type,
            "prompt_generator": self.prompt_generator,
            "command_registry": self.command_registry
        }
        return agent_dict
    

    @staticmethod
    def load(config_file: str = '') -> "AgentModel":
        
        """
        TODO DEPRECATED - THIS IS TO MAINTAIN BACKWARD COMPATIBILITY

        Returns class object with parameters (ai_name, ai_role, ai_goals, api_budget) loaded from
        yaml file if yaml file exists, else returns class with no parameters.

        Args:
            config_file (str, optional): The path to the config yaml file. Defaults to ''.

        Returns:
            cls (AgentModel): An instance of the `AgentModel` class.
        """

        try:
            with open(config_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
            ai_name = config_params.get("ai_name", "")
            ai_role = config_params.get("ai_role", "")
            ai_goals = config_params.get("ai_goals", [])
            api_budget = config_params.get("api_budget", 0.0)
            agent =  AgentModel(ai_name, ai_role, ai_goals, api_budget)
            # type: Type[AIConfig]

            ProjectsBroker.create(agent)

            # Move the file to the destination path
            #shutil.move(source_file, destination_path)

        except FileNotFoundError:
            ProjectsBroker.load()
        return 
