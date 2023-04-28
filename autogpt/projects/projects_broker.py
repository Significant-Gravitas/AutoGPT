"""
A module containing the ProjectConfigBroker class, which is used to manage the configuration settings for AI projects.

Description:
This module contains the ProjectConfigBroker class object which can be used to manage the configuration settings for AI projects. It provides methods to create a new project, set the current project, get the current project, get all the projects and get a specific project instance. It uses the Project and AgentConfig class objects from the autogpt.project module.

Functions:
None

Classes:
- ProjectConfigBroker:
A class object that contains the configuration information for the AI.

Global Variables:
- SAVE_FILE (str):
The path to the file where the configuration settings will be saved.
- MAX_AI_CONFIG (int):
The maximum number of AI configurations allowed.

Dependencies:
- os
- pathlib
- yaml
- shutil
- autogpt.prompts.generator.PromptGenerator
- autogpt.project.config.Project
- autogpt.project.agent.config.AgentConfig
- autogpt.singleton.AbstractSingleton
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Type, List
import shutil

import yaml

from autogpt.prompts.generator import PromptGenerator
from autogpt.projects.project import Project
from autogpt.projects.agent_model import AgentModel
from autogpt.singleton import AbstractSingleton


# Soon this will go in a folder where it remembers more stuff about the run(s)
SAVE_FILE = str(Path(os.getcwd()) / "agent_settings.yaml")
MAX_AI_CONFIG = 5


class ProjectsBroker(AbstractSingleton):   
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
        __version (str): Version of the app
        config_file (str): The path to the config yaml file.
    """

    def __init__(self, project_number: int = -1, config_file: str = None) -> None:
        """
    Initializes an ProjectConfigBroker instance with the specified project number and config file.

    Args:
        project_number (int, optional): The project number to be set as the current project. Defaults to -1.
        config_file (str, optional): The path to the config yaml file. Defaults to None.
    """
        self.config_file = config_file or SAVE_FILE
        self._current_project_id = project_number
        self._load(self.config_file)
        if 0 < project_number <= len(self._projects):
            self.set_project_number(project_number)

        shutil.copy(self.config_file, f"{self.config_file}.backup")

    def _load(self, config_file: str = SAVE_FILE) -> list:
        """
        Loads the projects from the specified YAML file and returns a list of Project instances containing the project parameters.

        Args:
            config_file (str): The path to the config yaml file. DEFAULT: "../agent_settings.yaml"
        
        Returns:
            list: A list of Project instances containing the project parameters for each entry.
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
                lead_agent = AgentModel(
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
                    delegated_agents = AgentModel(
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
        Saves the current configuration to the specified file yaml file path as a yaml file.

        Args:
            config_file (str): The path to the config yaml file. DEFAULT: "../agent_settings.yaml"
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
        Creates a new project with the specified parameters and adds it to the list of projects.

        Args:
            project_id (int): The project ID.
            api_budget (float): The maximum dollar value for API calls (0.0 means infinite).
            agent_name (str): The AI name.
            agent_role (str): The AI role.
            agent_goals (list): The AI goals.
            prompt_generator (str): The prompt generator instance.
            command_registry (str): The command registry instance.
            overwrite (bool, optional): Overwrite the project if it exists. Defaults to False.
            project_name (str, optional): The name of the project. Defaults to ''.

        Returns:
            bool: True if the project is successfully created, False otherwise.
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

        lead_agent = AgentModel(
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
        Sets the current project number.

        Args:
            new_project_id (int): The new project number to be set.

        Returns:
            bool: True if the project number is successfully set, False otherwise.
        """

        if new_project_id < 0 or new_project_id >= len(self._projects):
            raise ValueError(f"set_project_number: Value must be between 0 and {len(self._projects)-1}")
        self._current_project_id = new_project_id
        return True

    def get_current_project_id(self) -> int:
        """
        Gets the current project number.

        Parameters:
            None
        
        Returns:
            int: The current project number.
        """
        return self._current_project_id

    def get_current_project(self) -> Project:
        """
        Gets the current project instance.

        Parameters:
            None

        Returns:
            Project: The current project instance.
        """
        if self._current_project_id == -1:
            raise ValueError(f"get_current_project: Value {self._current_project_id } not expected")
        return self._projects[self._current_project_id]
    
    def get_project(self, project_number : int ) -> AgentModel:
        """
        Gets the specified project instance.

        Parameters:
            None

        Args:
            project_number (int): The project number.

        Returns:
            AgentConfig: The specified project instance.
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
    