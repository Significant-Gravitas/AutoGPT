"""
This module provides a class for configuring an agent's settings.
 
Description:
- This module contains the `AgentConfig` class, which represents the configuration settings for an agent.


Functions: None

Global Variables: None

Classes:
- AgentConfig: A class representing the configuration settings for an agent.

Dependencies: 
- typing: A built-in module for type hints and annotations.


"""


import os
import sys
import uuid
import yaml
from pathlib import Path
from typing import Optional, Type, List
# Soon this will go in a folder where it remembers more stuff about the run(s)
# @TODO 
SAVE_FILE = Path.cwd() / "ai_settings.yaml"
PROJECT_DIR = "autogpt/projects"
AUTOGPT_VERSION = 'X.Y.Z' # TODO, implement in config.py or main or technical env file



import abc
from autogpt.core.budget import BudgetManager
from autogpt.core.command import CommandRegistry
from autogpt.core.configuration import Configuration
from autogpt.core.llm import LanguageModel
from autogpt.core.logging import Logger
from autogpt.core.memory import MemoryBackend
from autogpt.core.messaging import MessageBroker
from autogpt.core.planning import Planner
from autogpt.core.plugin import PluginManager
from autogpt.core.workspace import Workspace



class AgentModel(): 
    """
    A class representing the configuration settings for an agent.

    Attributes:
        agent_name (str): The name of the agent.
        agent_goals (List[str]): A list of the agent's goals.
        agent_role (str): The role of the agent.
        agent_model (Optional[str]): The name of the agent's model, if applicable.
        agent_model_type (Optional[str]): The type of the agent's model, if applicable.
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
    
    # COMMENT ATTRIBUTES THAT SHOULD NOT BE SERIALIZED        
    NOT_SERIALIZE = ['NOT_SERIALIZE', 
                    #'configuration',
                    'logger',
                    #'budget_manager',
                    'command_registry',
                    #'language_model',
                    #'memory_backend',
                    #'message_broker',
                    #'planner',
                    #'plugin_manager',
                    #'workspace',
                    ]

    def __init__(self, 
                ai_name: str = '', #TODO PROJECT-Deprecated to make mandatory
                ai_role: str = '', #TODO PROJECT-Deprecated to make mandatory
                ai_goals: List = [], #TODO PROJECT-Deprecated to make mandatory
                agent_model: str = '', 
                agent_model_type: str = '',
                api_budget: float = 0.0, #TODO PROJECT-Deprecated to remove
                # team_name: Optional[str] = None,
                prompt_generator =  None,
                command_registry =  None,
                configuration: Configuration =  None,
                logger: Logger =  None,
                budget_manager: BudgetManager =  None,
                # command_registry: CommandRegistry =  None,
                language_model: LanguageModel =  None,
                memory_backend: MemoryBackend =  None,
                message_broker: MessageBroker =  None,
                planner: Planner =  None,
                plugin_manager: PluginManager =  None,
                workspace: Workspace =  None,
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
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals
        self.api_budget = api_budget #TODO PROJECT-Deprecated to remove
        self.prompt_generator= prompt_generator



        self.logger = Logger(),
        self.budget_manager= BudgetManager(),
        #self.command_registry= CommandRegistry(), 
        self.command_registry= command_registry  #TODO REARCH-Deprecated to remove
        self.language_model= LanguageModel(),
        self.memory_backend= MemoryBackend(),
        self.message_broker= MessageBroker(),
        self.planner= Planner(),
        self.plugin_manager= PluginManager(),
        self.workspace=  Workspace()

        self.uniq_id = self.generate_uniqid() # Generate a UUID)



    @classmethod
    def load(cls, config_file: str = SAVE_FILE) -> "AgentModel":
        from autogpt.config import Config
        CGF = Config()
        """
        TODO DEPRECATED - THIS IS TO MAINTAIN BACKWARD COMPATIBILITY


        Returns class object with parameters (ai_name, ai_role, ai_goals, api_budget) loaded from
        yaml file if yaml file exists, else returns class with no parameters.

        Args:
            config_file (str, optional): The path to the config yaml file. Defaults to ''.

        Returns:
            cls (AgentModel): An instance of the `AgentModel` class.
        """
        from autogpt.core.projects_broker import ProjectsBroker


        subfolders = [f.path for f in os.scandir(PROJECT_DIR) if f.is_dir() and f.name != '__pycache__']       
        if not subfolders:
            try:
                with open(config_file, encoding="utf-8") as file:
                    config_params = yaml.load(file, Loader=yaml.FullLoader)
            except FileNotFoundError:
                config_params = {}
            
            # FIXED LOAD DEPRECATED
            ai_name = config_params.get("ai_name", "")
            ai_role = config_params.get("ai_role", "")
            ai_goals = [
            str(goal).strip("{}").replace("'", "").replace('"', "")
            if isinstance(goal, dict)
            else str(goal)
            for goal in config_params.get("ai_goals", [])
            ]
            api_budget = config_params.get("api_budget", 0.0)
            agent =  cls(ai_name = ai_name, 
                                ai_role = ai_role, 
                                ai_goals = ai_goals, 
                                api_budget = api_budget)

            return agent
        else :  
            projects = ProjectsBroker().get_projects()
            return projects[0].agent_team.lead_agent
         
    @classmethod
    def new_load_agent(cls, agent_data: dict) -> "AgentModel":
        """
        Loads agent data from a dictionary and returns an `AgentModel` instance.

        Args:
            agent_data (dict): A dictionary containing the agent's data.

        Returns:
            agent_instance (AgentModel): An `AgentModel` instance with the loaded data.
        """
        agent_name = agent_data["agent_name"]
        agent_goals = [goal for goal in agent_data["agent_goals"]]
        agent_role = agent_data["agent_role"]
        agent_model = agent_data["agent_model"]
        agent_model_type = agent_data["agent_model_type"]

        load_dict = []
        # MODULAR LOAD ALLOW FOR PLUG IN
        for key, value in agent_data.items():
            # Check if _init_attribute_{key} method exists
            if hasattr(load_dict, f"_load_attribute_{key}"):
                # Call the method and pass the corresponding value
                setattr(load_dict, key, getattr(load_dict, f"_load_attribute_{key}")(value))

            # elif hasattr(agent, f"_savenload_attribute_{key}"):
            #     getattr(agent,  f"_savenload_attribute_{key}")(value)
                
            else:
                # Set the attribute on the instance
                setattr(load_dict, key, value)
        
        # return cls(agent_dictionary = load_dict)

        return  cls(
            ai_name=agent_name,
            ai_role=agent_role,
            ai_goals=agent_goals,
            agent_model=agent_model,
            agent_model_type=agent_model_type,
            config = Configuration()
            )
    
    def save(self, config_file: str = SAVE_FILE) -> dict:
        """
        Saves the `AgentModel` object as a dictionary representation.

        Returns:
            agent_dict (dict): A dictionary representation of the `AgentModel` object.
        """
        subfolders = [f.path for f in os.scandir(PROJECT_DIR) if f.is_dir() and f.name != '__pycache__']       
        if not subfolders:
            from autogpt.core.projects_broker import ProjectsBroker
            projects_broker = ProjectsBroker(config_file = config_file , do_not_load=True)
            project = projects_broker.create_project(
                    project_position_number = 0,
                    lead_agent = self,
                    project_budget = self.api_budget,
                    project_name = self.ai_name,
                    version = AUTOGPT_VERSION)
            return project.agent_team.lead_agent
        else :
            return self
    
    def new_save(self) :
        save_dict = {}
        for key, value in self.__dict__.items():
            # NOTE : UNCOMMENT THIS CODE TO AV
            if key in self.NOT_SERIALIZE :  
                 continue
            if hasattr(self, f"_save_attribute_{key}"):
                getattr(self, f"_save_attribute_{key}")()
            # elif hasattr(self, f"_savenload_attribute_{key}"):
            #     getattr(self, f"_savenload_attribute_{key}")()
            else:
                save_dict[key] = value
        print(save_dict)
        return save_dict


    def to_dict(self) -> dict:  
        agent_dict = {
                "agent_name": self.ai_name,
                "agent_role": self.ai_role,
                "agent_goals": self.ai_goals,
                "agent_model": self.agent_model,
                "agent_model_type": self.agent_model_type,
                "prompt_generator": self.prompt_generator,
                "command_registry": self.command_registry
            }
        
        save_dict = {}
        for key, value in self.__dict__.items():
            # if key.startswith("_"):  # Skip private attributes
            #     continue
            if hasattr(self, f"_save_attribute_{key}"):
                getattr(self, f"_save_attribute_{key}")()
            # elif hasattr(self, f"_savenload_attribute_{key}"):
            #     getattr(self, f"_savenload_attribute_{key}")()
            else:
                save_dict[key] = value
        print(save_dict)
        return agent_dict
    
    # A class to generate UUID so plugin ay overid it 
    # https://github.com/Significant-Gravitas/Auto-GPT/discussions/3392#step-2-discussed-features
    def generate_uniqid(self) -> uuid : 
        return str(uuid.uuid4())
    
    # All credit to pr-0f3t, I implemented it here, also I am not sure it shoudl be the only place
    def _load_attribute_ai_goals(self, value) :
        goals = []
        for goal in value : 
            if isinstance(goal, dict) :
                goals.append(str(goal).strip("{}").replace("'", "").replace('"', ""))
            else :
                goals.append(str(goal))

    # All credit to pr-0f3t, I implemented it here, also I am not sure it shoudl be the only place
    def _save_attribute_ai_goals(self, value) :
        goals = []
        for goal in value : 
            if isinstance(goal, dict) :
                goals.append(str(goal).strip("{}").replace("'", "").replace('"', ""))
            else :
                goals.append(str(goal))
        
        return goals


 
