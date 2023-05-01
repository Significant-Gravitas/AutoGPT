"""
A module containing the ProjectConfigBroker class, which is used to manage the configuration settings forprojects.

Description:
This module contains the ProjectConfigBroker class object which can be used to manage the configuration settings forprojects. It provides methods to create a new project, set the current project, get the current project, get all the projects and get a specific project instance. It uses the Project and AgentConfig class objects from the autogpt.project module.

Functions:
None

Classes:
- ProjectConfigBroker:
A class object that contains the configuration information for the AI.

Global Variables:
- SAVE_FILE (str):
The path to the file where the configuration settings will be saved.
- MAX_AI_CONFIG (int):
The maximum number of configurations allowed.

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

# Standard library imports
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Type, List

# Third-party imports
import yaml

# Local application imports
from autogpt.singleton import AbstractSingleton
from autogpt.projects.project import Project
from autogpt.projects.agent_model import AgentModel

# @TODO MOVE to .env
SAVE_FILE = Path.cwd() / "ai_settings.yaml"
PROJECT_DIR = "autogpt/projects"
MAX_NB_PROJECT = 1

class ProjectsBroker(AbstractSingleton):
    """
    The ProjectsBroker class is a singleton that manages multiple projects within AutoGPT. It is responsible for loading,
    saving, creating, and retrieving project configurations.

    Attributes:
        SAVE_FILE (Path): The path to the AutoGPT configuration file.
        PROJECT_DIR (str): The directory containing project folders.
        MAX_NB_PROJECT (int): The maximum number of projects allowed.
        AUTOGPT_VERSION (str): The current version of AutoGPT.

    Methods:
        __init__(self, config_file: str = None) -> None: Initializes a ProjectConfigBroker instance.
        load(self) -> list: Loads project configurations from the config file.
        _save(self, is_creation: bool = False, project_position_number: int = -1) -> None: Saves project configurations to the config file.
        create_project(self, project_position_number: int, lead_agent: AgentModel, project_budget: float, project_name: str = '', version: str = AUTOGPT_VERSION) -> Project: Creates a new project.
        project_dir_name_formater(project_name: str) -> str: Formats a project name for a valid directory name.
        create_project_dir(cls, project_name) -> str: Creates a new project directory.
        get_project_folder_list() -> list: Retrieves a list of project folder paths.
        set_project_positon_number(cls, new_project_position: int) -> bool: Sets the current project number.
        get_current_project_position(self) -> int: Retrieves the current project number.
        get_current_project(self) -> Project: Retrieves the currently selected project.
        get_project(self, project_positon_number: int) -> AgentModel: Retrieves a project by its position number.
        get_projects(self) -> list[Project]: Retrieves a list of all projects.
        is_loaded(self) -> bool: Checks if the configuration file has been loaded.
    """
    SAVE_FILE = SAVE_FILE
    PROJECT_DIR = PROJECT_DIR
    MAX_NB_PROJECT = MAX_NB_PROJECT
    AUTOGPT_VERSION = 'X.Y.Z' 
    def __init__(self,  
                 config_file: str = None,
                 do_not_load : bool = False # TODO Deprecated-Project
                 ) -> None:
        """
        Initializes a ProjectConfigBroker instance with the specified project number and config file.

        Args:
            config_file (str, optional): The path to the config yaml file. Defaults to None.
        """
        if not config_file == None or not hasattr(self, '_config_file') or self._config_file is None:
            self._config_file = config_file

        if(not hasattr(self, '_projects') or  self._projects == [] ) :
            self._current_project_position = -1
            self._projects = []
            if not do_not_load :
                self.load()

    def load(self) -> list:
        """
        Loads the projects from the specified YAML file and returns a list of Project instances containing the project parameters.

        Returns:
            list: A list of Project instances containing the project parameters for each entry.

        Raises:
            FileNotFoundError: If the specified config file does not exist.
            Exception: If no project folders are found.
        """

        self._projects = []
        projectfolder_list = ProjectsBroker.get_project_folder_list()
        if not projectfolder_list:
            raise Exception('ProjectsBroker.load() : Unexpected Behaviour')
        else :
            for i, projectfolder in enumerate(projectfolder_list) :
                configfile = os.path.join(os.path.abspath(projectfolder),'settings.yaml') 
                if not os.path.exists(configfile):
                    print('ProjectsBroker.load() : Unexpected setting.yaml missing')
                    continue

                # NOTE : NOT REALLY REQUIRED BUT PUTTING EVERY SAFEGUARD FOR BACKWARD COMPATIBILITY
                if i < MAX_NB_PROJECT : 
                    with open(configfile, encoding="utf-8") as file:
                        config_params = yaml.load(file, Loader=yaml.SafeLoader)

                        project_instance = Project.load(config_params)
                        self._projects.append(project_instance)
                    i += 1

        return self._projects

    def _save(self, is_creation = False, project_position_number: int = -1) -> None:
        """
        Saves the current state of the ProjectsBroker to the configuration file if a project_position_number is passed. Else, it will save all the projects.

        Args:
            is_creation (bool, optional): Whether the project is being created or not. Defaults to False.
            project_position_number (int, optional): The position of the project to save. Defaults to -1.

        Raises:
            IndexError: If the project_position is out of range.
        """
        
        if 0 <= project_position_number < len(self._projects):
            self._projects[project_position_number].save( 
                project_position_number = project_position_number, 
                is_creation = is_creation)
        elif project_position_number == -1:
            # Save all projects if project_position is not specified
            for project in self._projects:
                project.save()
        else:
            raise IndexError("Project index out of range.")
 
    
    from autogpt.projects.agent_model import AgentModel
   
    def create_project(self, 
                    project_position_number : int,
                    lead_agent : AgentModel,
                    project_budget : float,
                    project_name : str = '',
                    version : str = AUTOGPT_VERSION
                    ) -> Project:
        """
        Creates a new project with the specified parameters and adds it to the list of projects.

        Args:
            project_position_number (int): The project ID.
            lead_agent (AgentModel): The lead agent for the project.
            project_budget (float): The maximum dollar value for API calls (0.0 means infinite).
            project_name (str, optional): The name of the project. Defaults to ''.
            version (str, optional): The version of AutoGPT when the project was last run. Defaults to ProjectsBroker.AUTOGPT_VERSION.


        Returns:
            AgentModel: The lead agent instance for the new project.

        Raises:
            ValueError: If the project_position is greater than the maximum number of projects allowed.
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
                          project_budget= project_budget,
                          lead_agent= lead_agent,
                          version = version)
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

        self.set_project_positon_number(new_project_position=project_position_number)
        self._save(project_position_number=project_position_number ,is_creation = is_creation)

        
        return project
    
    @staticmethod   
    def project_dir_name_formater(project_name : str ) -> str :
        """
        Formats a project name as a valid directory name by removing non-alphanumeric characters and converting to lowercase.

        Parameters:
            project_name (str): The name of the project to be formatted.

        Returns:
            str: The formatted project name.
        """
        return re.sub(r'[^a-zA-Z0-9]', '', project_name).lower()

    @classmethod   
    def create_project_dir(cls, project_name) -> str : 
        """
        Creates a directory for a new project using the specified name.

        Parameters:
            project_name (str): The name of the new project.

        Returns:
            str: The path to the new project directory.
        """
        project_dirname = cls.project_dir_name_formater(project_name)
        os.mkdir(os.path.join(PROJECT_DIR, project_dirname))

    @staticmethod  
    def get_project_folder_list() -> list : 
        """
        Gets the list of project folders from the 'autogpt/projects' directory.

        Parameters:
            None

        Returns:
            list: A list of folder paths for each project.
        """
        return [f.path for f in os.scandir(PROJECT_DIR) if f.is_dir() and f.name != '__pycache__' and not f.name.endswith('.backup') ]     
        

    @classmethod   
    def set_project_positon_number(cls,  new_project_position: int) -> bool:
        """
        Sets the current project number.

        Args:
            new_project_position (int): The new project number to be set.

        Returns:
            bool: True if the project number is successfully set, False otherwise.
        """
        projects_broker = cls()
        if new_project_position < 0 or new_project_position >= len(projects_broker._projects):
            raise ValueError(f"set_project_positon_number: Value must be between 0 and {len(projects_broker._projects)-1}")
        projects_broker._current_project_position = new_project_position
        return True

    def get_current_project_position(self) -> int:
        """
        Gets the current project number.

        Parameters:
            None
        
        Returns:
            int: The current project number.
        """
        return self._current_project_position

    def get_current_project(self) -> Project:  
        """
        Gets the currently selected Project instance in the ProjectsBroker instance.

        Returns:
            Project: The currently selected Project instance.

        Raises:
            ValueError: If there is no currently selected project.
        """
        if self._current_project_position == -1:
            raise ValueError(f"get_current_project: Value {self._current_project_position } not expected")
        return self._projects[self._current_project_position]
    
    def get_project(self, project_positon_number : int ) -> AgentModel:
        """
        Gets the Project instance with the specified position number in the ProjectsBroker instance.

        Args:
            project_positon_number (int): The position number of the project to retrieve.

        Returns:
            Project: The Project instance with the specified position number.
        
        Raises:
            ValueError: If the specified project position number is out of range.
        """
        if project_positon_number < 0 or len(self._projects) < project_positon_number  :
            raise ValueError(f"get_config: Value {project_positon_number } not expected")     
        return self._projects[project_positon_number]

 

    def get_projects(self) -> list[Project]:
        """
        Gets the Project instance with the specified position number in the ProjectsBroker instance.

        Args:
            project_positon_number (int): The position number of the project to retrieve.

        Returns:
            Project: The Project instance with the specified position number.
        
        Raises:
            ValueError: If the specified project position number is out of range.
        """
        return self._projects
    
    def is_loaded(self) -> bool : 
        """
        Checks if the configuration file has been loaded.

        Returns:
            bool: True if the configuration file has been loaded, False otherwise.
        """

        if hasattr(self , '_projects') and len(self._projects > 0) :
            return False
        else : 
            return True
    

    