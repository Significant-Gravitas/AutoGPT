# sourcery skip: do-not-use-staticmethod
"""
A module that contains the AIConfig class object that contains the configuration
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Type, List
import shutil

import yaml

from autogpt.prompts.generator import PromptGenerator

# Soon this will go in a folder where it remembers more stuff about the run(s)
SAVE_FILE = str(Path(os.getcwd()) / "agent_settings.yaml")
MAX_AI_CONFIG = 5

from .singleton import AbstractSingleton

class AIConfigBroker(AbstractSingleton):   
    """
    A class object that contains the configuration information for the AI

    Attributes:
        __current_project (int): The number of the current project from 0 to 4.
        __projects (list): A list of Dictionary (projects) :
            {"agent_name": agent_name, 
            "agent_role": agent_role, 
            "agent_goals": agent_goals, 
            "prompt_generator": prompt_generator,
            "command_registry": command_registry 
            api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
            })
        # TODO __version (str): Version of the app
    """

    def __init__(self, project_number: int = -1, config_file: str = None) -> None:
        """_summary_

        Args:
            project_number (int, optional): _description_. Defaults to -1.
            config_file (str, optional): _description_. Defaults to None.
        """
        self.config_file = config_file or SAVE_FILE
        self._current_project_id = project_number
        self._load(self.config_file)
        if 0 < project_number <= len(self._projects):
            self.set_project_number(project_number)

        shutil.copy(self.config_file, f"{self.config_file}.backup")

    def _load(self, config_file: str = SAVE_FILE) -> list:
        """
        Loads the projects from the specified YAML file and returns a list of dictionaries containing the project parameters.

        Parameters:
            config_file(str): The path to the config yaml file.
            DEFAULT: "../agent_settings.yaml"
        
        Returns:
            agent_configs (list): A list of dictionaries containing the project parameters for each entry.
        """
        
        if not os.path.exists(config_file):
            self._projects = []
            return self._projects

        with open(config_file, encoding="utf-8") as file:
            config_params = yaml.load(file, Loader=yaml.SafeLoader)

        self._projects = []
        version =  config_params.get("version", '')
        if version != '' :
            self._version = version # Not supported for the moment
        for project in config_params.get("projects", []):
            if (project.get("project_name")) :
                project_name = project["project_name"]
            else :
                raise ValueError("No project_name in the project.")

            if (project.get("budget")) :
                api_budget = project["budget"]
            else :
                raise ValueError("No budget in the project.")

            if (project.get("lead_agent")) :
                lead_agent_data = project["lead_agent"]
                lead_agent = AgentConfig(
                    agent_name=lead_agent_data["agent_name"],
                    agent_role=lead_agent_data["agent_role"],
                    agent_goals=lead_agent_data["agent_goals"],
                    agent_model=lead_agent_data.get("agent_model", None),
                    
                )
            else :
                raise ValueError("No lead_agent in the project.")

                
            delegated_agents_list = []
            if (project.get("delegated_agents")) :
                for delegated_agents_data in project["delegated_agents"]:
                    delegated_agents = AgentConfig(
                        agent_name=delegated_agents_data["agent_name"],
                        agent_role=delegated_agents_data["agent_role"],
                        agent_goals=delegated_agents_data["agent_goals"],
                        agent_model=delegated_agents_data.get("agent_model", None),
                        agent_model_type=delegated_agents_data.get("agent_model_type", None),
                    )
                    delegated_agents_list.append(delegated_agents)

            self._projects.append(Project(project_name =  project_name, api_budget =  api_budget,lead_agent = lead_agent, delegated_agents = delegated_agents_list))
            
            #break 
            """
            # 
            This break allows to push the new YAML back-end
            A PR with thisbreak will not allow multiple projects but set the new architecture for multiple model
            """

        return self._projects

    def _save(self , config_file: str = SAVE_FILE) -> None:
        """
        Saves the current projecturation to the specified file yaml file path as a yaml file.

        Parameters:
            config_file(str): The path to the config yaml file.
            DEFAULT: "../agent_settings.yaml"
        Returns:
            None
        """
  
        if not os.path.exists(config_file):
            with open(config_file, "w", encoding="utf-8") as file:
                yaml.dump({"projects": []}, file, allow_unicode=True)
                
        try:
            with open(config_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.SafeLoader)
        except FileNotFoundError:
            config_params = {}

        current_project_str = self.get_current_project().to_dict()

        if "projects" not in config_params:
            config_params["projects"] = []

        if 0 <= self._current_project_id < len(config_params["projects"]):
            config_params["projects"][self._current_project_id] = current_project_str
        else:
            config_params["projects"].append(current_project_str)

        # data_to_save = {"version": self._version, "projects": config_params}
        data_to_save = config_params
        with open(config_file, "w", encoding="utf-8") as file:
            yaml.dump(data_to_save, file, allow_unicode=True)

 

    # def _save(self, config_file: str = SAVE_FILE) -> None:
    #     """
    #     Saves the current projecturation to the specified file yaml file path as a yaml file.

    #     Parameters:
    #         config_file(str): The path to the config yaml file.
    #         DEFAULT: "../agent_settings.yaml"
    #     Returns:
    #         None
    #     """
    #     project_data = []
    #     for project in self._projects:
    #         delegated_agents_data = []
    #         for delegated_agents in project.delegated_agents:
    #             delegated_agents_data.append(
    #                 {
    #                     "agent_name": delegated_agents.agent_name,
    #                     "agent_role": delegated_agents.agent_role,
    #                     "agent_goals": delegated_agents.agent_goals,
    #                     "agent_model": delegated_agents.agent_model,
    #                     "agent_model_type": delegated_agents.agent_model_type,
    #                 }
    #             )
    #         project_data.append(
    #             {
    #                 "project_name": project.project_name,
    #                 "lead_agent": {
    #                     "agent_name": project.lead_agent.agent_name,
    #                     "agent_role": project.lead_agent.agent_role,
    #                     "agent_goals": project.lead_agent.agent_goals,
    #                     "agent_model": project.lead_agent.agent_model,
    #                     "agent_model_type": project.lead_agent.agent_model_type,
    #                 },
    #                 "delegated_agents": delegated_agents_data,
    #             }
    #         )

    #     data_to_save = {"version": "X.Y.X", "projects": project_data}
    #     with open(config_file, "w", encoding="utf-8") as file:
    #         yaml.dump(data_to_save, file, allow_unicode=True)

    def create_project(self, 
                    project_id : int,
                    api_budget :float,  
                    agent_name : str, 
                    agent_role : str, 
                    agent_goals : list, 
                    prompt_generator : str, 
                    command_registry : str ,
                    overwrite : bool = False,
                    project_name : str = '' ) -> bool:
        """
        Sets the project parameters for the given configuration number.

        Parameters:
            project_number (int): The project number to be updated.
            agent_name (str): The AI name.
            agent_role (str): The AI role.
            agent_goals (list): The AI goals.
            prompt_generator (Optional[PromptGenerator]): The prompt generator instance.
            command_registry: The command registry instance.

        Returns:
            success (bool): True if the configuration is successfully updated, False otherwise.
        """
        project_name = agent_name # TODO : Allow to give a project name :-)

        number_of_config = len(self._projects)
        if project_id is None and number_of_config < MAX_AI_CONFIG:
            project_id = number_of_config

        if project_id > MAX_AI_CONFIG:
            raise ValueError(f"set_config: Value {project_id} not expected")

        # Check for duplicate config names
        for idx, config in enumerate(self._projects):
            if config.project_name == project_name and idx != project_id:
                print(f"Config with the name '{project_name}' already exists")
                return False

        lead_agent = AgentConfig(
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            prompt_generator=prompt_generator,
            command_registry=command_registry)
        
        project = Project(project_name= project_name , api_budget=api_budget,lead_agent= lead_agent )
        
        if project_id >= number_of_config:
            self._projects.append(project)
            project_id = number_of_config
        else:
            self._projects[project_id] = project


        self.set_project_number(new_project_id=project_id)
        self._save()
        
        return True
           
    def set_project_number(self, new_project_id: int) -> bool:
        """
        Sets the current projecturation number.

        Parameters:
            new_project_number (int): The new project number to be set.

        Returns:
            None
        """
        if new_project_id < 0 or new_project_id >= len(self._projects):
            raise ValueError(f"set_project_number: Value must be between 0 and {len(self._projects)-1}")
        self._current_project_id = new_project_id
        return True

    def get_current_project_id(self) -> int:
        """
        Gets the current projecturation number.

        Parameters:
            None

        Returns:
            __current_project_id (int): The current projecturation number.
        """
        return self._current_project_id

    def get_current_project(self) -> Project:
        """
        Gets the current project dictionary.

        Parameters:
            None

        Returns:
            current_project (dict): The current project dictionary.
        """
        if self._current_project_id == -1:
            raise ValueError(f"get_current_project: Value {self._current_project_id } not expected")
        return self._projects[self._current_project_id]
    
    def get_project(self, project_number : int ) -> AgentConfig:
        """
        Gets the current project dictionary.

        Parameters:
            None

        Returns:
            current_project (dict): The current project dictionary.
        """
        if 0 <= project_number <= len(self._projects):
            raise ValueError(f"get_config: Value {project_number } not expected")     
        return self._projects[project_number]

 

    def get_projects(self) -> list:
        """
        Gets the list of all configuration AIConfig.

        Parameters:
            None

        Returns:
            configs (list): A list of all project AIConfig.
        """

        return self._projects
    
class Project:
    def __init__(self, project_name : str, api_budget : float , lead_agent : AgentConfig, delegated_agents : List[AgentConfig] = []):
        self.project_name = project_name
        self.api_budget= api_budget
        self.lead_agent = lead_agent
        self.delegated_agents = delegated_agents
    
    #saving in Yaml
    def __str__(self) -> str:

        return str(self.to_dict())
    
    def __toDict__(self) -> dict :

        return self.to_dict()
    
    def to_dict(self) -> dict:
        lead_agent_dict = {
            "agent_name": self.lead_agent.agent_name,
            "agent_role": self.lead_agent.agent_role,
            "agent_goals": self.lead_agent.agent_goals,
            "agent_model": self.lead_agent.agent_model,
            "agent_model_type": self.lead_agent.agent_model_type,
            "prompt_generator": self.lead_agent.prompt_generator,
            "command_registry": self.lead_agent.command_registry
        }
        delegated_agents_list = []
        for agent in self.delegated_agents:
            agent_dict = {
                "agent_name": agent.agent_name,
                "agent_role": agent.agent_role,
                "agent_goals": agent.agent_goals,
                "agent_model": agent.agent_model,
                "agent_model_type": agent.agent_model_type,
                "prompt_generator": agent.prompt_generator,
                "command_registry": agent.command_registry
            }
            delegated_agents_list.append(agent_dict)
        dict_representation = {
            "project_name": self.project_name,
            "api_budget": self.api_budget,
            "lead_agent": lead_agent_dict,
            "delegated_agents": delegated_agents_list
        }
        return dict_representation

    


        
class AgentConfig(): 
    """_summary_
    """
    def __init__(self, 
            agent_name: str,
            agent_role: str,
            agent_goals: List,
            agent_model: Optional[str] = None,
            agent_model_type: Optional[str] = None,
            prompt_generator =  None,
            command_registry =  None) -> None:
        """_summary_

        Args:
            agent_name (_type_): _description_
            agent_role (_type_): _description_
            agent_goals (_type_): _description_
            prompt_generator (_type_): _description_
            command_registry (_type_): _description_
        """

        self.agent_name = agent_name
        self.agent_role = agent_role
        self.agent_goals = agent_goals
        self.agent_model = agent_model
        self.agent_model_type = agent_model_type
        self.prompt_generator= prompt_generator
        self.command_registry= command_registry
