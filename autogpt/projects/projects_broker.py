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
import shutil
import re
import yaml
from autogpt.singleton import AbstractSingleton
from pathlib import Path
from typing import Optional, Type, List
from autogpt.prompts.generator import PromptGenerator

import sys
if not 'autogpt.projects.agent_model' in sys.modules:
    from autogpt.projects.agent_model import AgentModel
if not 'autogpt.projects.project' in sys.modules: 
    from autogpt.projects.project import Project

# Soon this will go in a folder where it remembers more stuff about the run(s)
# @TODO 
SAVE_FILE = str(Path(os.getcwd()) / "ai_settings.yaml")
PROJECT_DIR = "autogpt/projects"
MAX_NB_PROJECT = 5

class ProjectsBroker(AbstractSingleton):   
    """
    A singleton class that manages the configuration settings for AI projects.

    This class contains the ProjectsBroker object, which can be used to create, manage, and save configurations for AI projects. It provides methods to create a new project, set the current project, get the current project, get all the projects, and get a specific project instance. The class depends on the Project and AgentConfig class objects from the autogpt.project module and the AbstractSingleton class from the autogpt.singleton module.

    Attributes:
        SAVE_FILE (str): The path to the file where the configuration settings will be saved.
        MAX_NB_PROJECT (int): The maximum number of AI configurations allowed.
        __current_project_id (int): The number of the current project from 0 to 4.
        __projects (list): A list of dictionaries containing project information, such as the lead agent, project budget, and prompt generator.
        __version (str): Version of the app.
        config_file (str): The path to the config yaml file.

    Methods:
        __init__(self, project_positon_number: int = -1, config_file: str = None) -> None:
            Initializes a ProjectsBroker instance with the specified project number and config file.

        load(self, config_file: str = '') -> list:
            Loads the projects from the specified YAML file and returns a list of Project instances containing the project parameters.

        _save(self, project_position: int = -1) -> None:
            Saves the current state of the ProjectsBroker to the configuration file.

        create_project(self, project_id: int, lead_agent: AgentModel, api_budget: float, project_name: str = '') -> AgentModel:
            Creates a new project with the specified parameters and adds it to the list of projects.

        project_dir_name_formater(project_name: str) -> str:
            Formats a project name as a directory name.

        project_dir_create(cls, project_position: str,  project_name: str) -> str:
            Creates a directory for a project.

        set_project_positon_number(self, new_project_id: int) -> bool:
            Sets the current project number.

        get_current_project_id(self) -> int:
            Gets the current project number.

        get_current_project(self) -> Project:
            Gets the current project instance.

        get_project(self, project_positon_number: int) -> AgentModel:
            Gets the specified project instance.

        get_projects(self) -> list:
            Gets the list of all projects.
    """

   # def __init__(self, project_positon_number: int = -1, config_file: str = None) -> None:
    #     """
    #     Initializes a ProjectConfigBroker instance with the specified project number and config file.

    #     Args:
    #         project_positon_number (int, optional): The project number to be set as the current project. Defaults to -1.
    #         config_file (str, optional): The path to the config yaml file. Defaults to None.
    #     """
    #     self.config_file = config_file 
    #     self._current_project_id = project_positon_number
    #     self.load(self.config_file)
    #     if 0 <= project_positon_number <= len(self._projects):
    #         self.set_project_positon_number(project_positon_number)

        # TODO create the back-up at a project level
    #     shutil.copy(self.config_file, f"{self.config_file}.backup")

    def __init__(self,  config_file: str = None) -> None:
        """
        Initializes a ProjectConfigBroker instance with the specified project number and config file.

        Args:
            project_positon_number (int, optional): The project number to be set as the current project. Defaults to -1.
            config_file (str, optional): The path to the config yaml file. Defaults to None.
        """
        if not config_file == None or not hasattr(self, '_config_file') or self._config_file is None:
            self._config_file = config_file
        self.config_file = config_file 
        self._current_project_id = -1
        self._projects = []


    @classmethod # TODO , to maintain backward compatibility but to be removed
    def load(cls, config_file: str = '') -> list:
        """
        Loads the projects from the specified YAML file and returns a list of Project instances containing the project parameters.

        Args:
            config_file (str): The path to the config yaml file.

        Returns:
            list: A list of Project instances containing the project parameters for each entry.

        Raises:
            FileNotFoundError: If the specified config file does not exist.
        """
        
        if not os.path.exists(config_file):
            cls._projects = []
            return cls._projects

        with open(config_file, encoding="utf-8") as file:
            config_params = yaml.load(file, Loader=yaml.SafeLoader)

        cls._projects = []
        for i , project_data in enumerate(config_params.get("projects", [])):
            project_instance = Project.load(project_data)
            cls._projects[i] = project_instance

        return cls._projects

    # def _save(self , config_file: str = '') -> None:
    #     """
    #     Saves the current configuration to the specified file yaml file path as a yaml file.

    #     Args:
    #         config_file (str): The path to the config yaml file.
    #     """
  
    #     if not os.path.exists(config_file):
    #         with open(config_file, "w", encoding="utf-8") as file:
    #             yaml.dump({"projects": []}, file, allow_unicode=True)
                
    #     try:
    #         with open(config_file, encoding="utf-8") as file:
    #             config_params = yaml.load(file, Loader=yaml.SafeLoader)
    #     except FileNotFoundError:
    #         config_params = {}

    #     current_project_str = self.get_current_project().to_dict()

    #     if "projects" not in config_params:
    #         config_params["projects"] = []

    #     if 0 <= self._current_project_id < len(config_params["projects"]):
    #         config_params["projects"][self._current_project_id] = current_project_str
    #     else:
    #         config_params["projects"].append(current_project_str)

    #     # data_to_save = {"version": self._version, "projects": config_params}
    #     data_to_save = config_params
    #     with open(config_file, "w", encoding="utf-8") as file:
    #        # yaml.dump(data_to_save, file, allow_unicode=True) @TODO enable
    #        return

    def _save(self, is_creation = False, project_position_number: int = -1) -> None:
        """
        Saves the current state of the ProjectsBroker to the configuration file if a project_position_number is passe. Else, it will save all the projects.
        # NOTE MAY BE INTERESSANT TO SPIT WITH SAVE ALL & make Project_positionnumber mandatory
        Args:
            project_position (int, optional): The position of the project to save. Defaults to -1.


        Raises:
            IndexError: If the project_position is out of range.

        """
        if 0 <= project_position_number < len(self._projects):
            """
            in Project.save()
            self.project_dir_create(project_position_number, self._projects[project_position_number].project_name)
            """
            self._projects[project_position_number].save( 
                project_position_number = project_position_number, 
                is_creation = is_creation)
        elif project_position_number == -1:
            # Save all projects if project_position is not specified
            for project in self._projects:
                project.save()
        else:
            raise IndexError("Project index out of range.")
 
    
    def create_project(self, 
                    project_position_number : int,
                    lead_agent : AgentModel,
                    api_budget : float,
                    project_name : str = '',
                    ) -> Project:
        """
        Creates a new project with the specified parameters and adds it to the list of projects.

        Args:
            project_id (int): The project ID.
            lead_agent : AgentModel: The lead agent for the project.
            api_budget (float): The maximum dollar value for API calls (0.0 means infinite).
            project_name (str, optional): The name of the project. Defaults to ''.

        Returns:
            AgentModel: The lead agent instance for the new project.

        Raises:
            ValueError: If the project_id is greater than the maximum number of projects allowed.
        """
        self = ProjectsBroker()
        # # REQUIRED FOR BACKWARD COMPATIBILITY , WILL NEED TO BE REMOVED ON LO
        # if not self.is_loaded():
        #     print('_projects')


        project_name = lead_agent.ai_name # TODO : Allow to give a project MIUST BE ALPHANUMERICAL + SPACE

        number_of_config = len(self._projects)
        if project_position_number is None and number_of_config < MAX_NB_PROJECT:
            project_position_number = number_of_config

        if project_position_number > MAX_NB_PROJECT:
            raise ValueError(f"set_config: Value {project_position_number} not expected")
        
        project = Project(project_name= project_name ,
                          project_budget= api_budget,
                          lead_agent= lead_agent
                          )
        # project_name: str, 
        #          project_budget: float, 
        #          lead_agent: AgentModel,
        #          delegated_agents: List[AgentModel] = [], 
        #          version: str = "0.0.0",
        #          project_memory: Optional[str] = None, 
        #          project_working_directory: Optional[str] = None,
        #          project_env: Optional[str] = None, 
        #          project_log_activity: Optional[str] = None,
        #          project_log_env: Optional[str] = None, 
        #          team_name: Optional[str] = None

        # TODO REMOVE is_creation ?
        is_creation = False
        if project_position_number >= number_of_config:
            self._projects.append(project)
            project_position_number = number_of_config
            is_creation = True
        else:
            self._projects[project_position_number] = project

        self.set_project_positon_number(new_project_id=project_position_number)
        self._save(project_position_number=project_position_number ,is_creation = is_creation)

        
        return project
    
    @staticmethod   
    def project_dir_name_formater( project_name : str ) -> str :
        return re.sub(r'[^a-zA-Z0-9]', '', project_name).lower()

    @classmethod   
    def create_project_dir(cls, project_name) -> str : 
        project_dirname = cls.project_dir_name_formater(project_name)
        os.mkdir(os.path.join(PROJECT_DIR, project_dirname))


    @classmethod   
    def set_project_positon_number(cls,  new_project_id: int) -> bool:
        """
        Sets the current project number.

        Args:
            new_project_id (int): The new project number to be set.

        Returns:
            bool: True if the project number is successfully set, False otherwise.
        """
        projects_broker = cls()
        if new_project_id < 0 or new_project_id >= len(projects_broker._projects):
            raise ValueError(f"set_project_positon_number: Value must be between 0 and {len(projects_broker._projects)-1}")
        projects_broker._current_project_id = new_project_id
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
    
    def get_project(self, project_positon_number : int ) -> AgentModel:
        """
        Gets the specified project instance.

        Parameters:
            None

        Args:
            project_positon_number (int): The project number.

        Returns:
            AgentConfig: The specified project instance.
        """
        if 0 <= project_positon_number <= len(self._projects):
            raise ValueError(f"get_config: Value {project_positon_number } not expected")     
        return self._projects[project_positon_number]

 

    def get_projects(self) -> list:
        """
        Gets the list of all configuration AIConfig.

        Parameters:
            None

        Returns:
            configs (list): A list of all project AIConfig.
        """

        return self._projects
    
    def is_loaded(self) -> bool : 
        if hasattr(self , '_projects') and len(self._projects > 0) :
            return False
        else : 
            return True
    

    