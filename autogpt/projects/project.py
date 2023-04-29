"""
This module provides a class representing an AI project with a name, API budget, lead agent, and delegated agents.
 
Classes:
- Project: A class representing an AI project with a name, API budget, lead agent, and delegated agents.

Functions:
- None

Global Variables:
- None

Dependencies:
- typing: A built-in module for type hints and annotations.
- AgentConfig: A class representing the configuration settings for an AI agent.

Attributes:
- project_name (str): The name of the project.
- api_budget (float): The budget allocated for using the OpenAI API.
- lead_agent (AgentConfig): The lead agent for the project.
- delegated_agents (List[AgentConfig]): A list of delegated agents for the project.

Methods:
- __init__(self, project_name: str, api_budget: float, lead_agent: AgentConfig, delegated_agents: List[AgentConfig] = []): Initializes a new Project object with the given parameters.
- __str__(self) -> str: Returns a string representation of the Project object.
- __toDict__(self) -> dict: Returns a dictionary representation of the Project object.
- to_dict(self) -> dict: Converts the Project object to a dictionary representation.
"""

import os
import shutil
import yaml
from typing import Optional, Type, List
import datetime

from autogpt.projects.agent_model import AgentModel

AUTOGPT_VERSION = 'X.Y.Z' # TODO, implement in config.py or main or technical env file
PROJECT_DIR = "autogpt/projects"

class Project:
    """
    A class representing an AI project with a name, API budget, lead agent, and delegated agents.

    Attributes:
        project_name (str): The name of the project.
        project_budget (float): The budget allocated for using the OpenAI API.
        lead_agent (AgentModel): The lead agent for the project.
        delegated_agents_list (List[AgentModel]): A list of delegated agents for the project.
        version (str): The version of the project.
        project_memory (Optional[str]): The memory used by the project.
        project_working_directory (Optional[str]): The working directory of the project.
        project_env (Optional[str]): The environment of the project.
        project_log_activity (Optional[str]): The log activity of the project.
        project_log_env (Optional[str]): The log environment of the project.

    Methods:
        __init__(self, project_name: str, project_budget: float, lead_agent: AgentModel,
                 delegated_agents_list: List[AgentModel] = [], version: str = "0.0.0",
                 project_memory: Optional[str] = None, project_working_directory: Optional[str] = None,
                 project_env: Optional[str] = None, project_log_activity: Optional[str] = None,
                 project_log_env: Optional[str] = None, team_name: Optional[str] = None) -> None
            Initializes a new Project object with the given parameters.
        load(cls, config_params: dict) -> "Project"
            Loads a portion of the configuration parameters and returns a Project instance.
        save(self, is_creation: bool = False, creation_position: int = -1) -> dict
            Saves the Project object as a dictionary representation.
        get_lead(self) -> AgentModel
            Returns the lead agent of the project.
        get_delegated_agents_list(self) -> List[AgentModel]
            Returns the list of delegated agents for the project.
        delete_delegated_agents(self, position: int) -> bool
            Deletes the delegated agent at the given position and returns True if successful.
    """
    def __init__(self, project_name: str, 
                 project_budget: float, 
                 lead_agent: AgentModel,
                 delegated_agents: List[AgentModel] = [],
                 version = AUTOGPT_VERSION,
                 project_memory: Optional[str] = None, 
                 project_working_directory: Optional[str] = None,
                 project_env: Optional[str] = None, 
                 project_log_activity: Optional[str] = None,
                 project_log_env: Optional[str] = None, 
                 team_name: Optional[str] = None):    
        """
        Initializes a new Project object with the given parameters.

        Args:
            project_name (str): The name of the project.
            project_budget (float): The budget allocated for using the OpenAI API.
            lead_agent (AgentModel): The lead agent for the project.
            delegated_agents_list (List[AgentModel], optional): A list of delegated agents for the project. Defaults to an empty list.
            version (str, optional): The version of the project. Defaults to "0.0.0".
            project_memory (str, optional): The memory used by the project. Defaults to None.
            project_working_directory (str, optional): The working directory of the project. Defaults to None.
            project_env (str, optional): The environment of the project. Defaults to None.
            project_log_activity (str, optional): The log activity of the project. Defaults to None.
            project_log_env (str, optional): The log environment of the project. Defaults to None.
            team_name (str, optional): The name of the team. Defaults to None.
        """
        self.version = version
        self.project_name = project_name
        self.project_budget = project_budget
        self.project_memory = project_memory
        self.project_working_directory = project_working_directory
        self.project_env = project_env
        self.project_log_activity = project_log_activity
        self.project_log_env = project_log_env
        self.team_name = team_name
        self.lead_agent = lead_agent
        self.delegated_agents_list = delegated_agents
    
    #saving in Yaml
    def __str__(self) -> str:

        return str(self.to_dict())
    
    def __toDict__(self) -> dict :

        return self.to_dict()
    
    def to_dict(self) -> dict:
        """
        Converts the Project object to a dictionary representation.

        Returns:
            dict_representation (dict): A dictionary representation of the Project object.
        """
        lead_agent_dict = self.lead_agent.to_dict() 

        delegated_agents_list = []
        for agent in self.delegated_agents_list:
            agent_dict = agent.to_dict() 
            delegated_agents_list.append(agent_dict)

        dict_representation = {
            "project_name": self.project_name,
            "project_budget": self.project_budget,
            "lead_agent": lead_agent_dict,
            "delegated_agents": delegated_agents_list
        }
        return dict_representation

    @classmethod
    def load(cls, config_params: dict) -> "Project":
        """
        Loads a portion of the configuration parameters and returns a Project instance.

        Args:
            config_params (dict): A dictionary containing the configuration parameters for the project.

        Returns:
            project_instance (Project): A Project instance with the loaded configuration parameters.
        """

        project_memory = config_params.get("project_memory")
        project_working_directory = config_params.get("project_working_directory")
        project_env = config_params.get("project_env")
        project_log_activity = config_params.get("project_log_activity")
        project_log_env = config_params.get("project_log_env")
        agent_team = config_params.get("agent_team", None)
        # team_name = agent_team.get("team_name")
        version =  config_params.get("version", AUTOGPT_VERSION)
        if version != '' :
            cls._version = version 
            # Not supported for the moment

        if (config_params.get("project_name")) :
            project_name = config_params["project_name"]
        else :
            raise ValueError("No project_name in the project.")

        if (config_params.get("project_budget")) :
            project_budget = config_params["project_budget"]
        else :
            raise ValueError("No budget in the project.")

        if config_params.get("lead_agent"):
            lead_agent_data = agent_team["lead_agent"]
            lead_agent = AgentModel.load(lead_agent_data)
        else:
            raise ValueError("No lead_agent in the project.")
       

        delegated_agents_list = []
        if agent_team.get("delegated_agents"):
            for delegated_agent_data in agent_team["delegated_agents"]:
                delegated_agent = AgentModel.load(delegated_agent_data)
                delegated_agents_list.append(delegated_agent)

        return cls(project_name, project_budget, lead_agent, delegated_agents_list,
                   version, project_memory, project_working_directory, project_env,
                   project_log_activity, project_log_env)
    

    def save(self, project_position_number : int, is_creation : bool = False  ) -> dict:
        """
        Saves the Project object as a dictionary representation.

        Args:
            is_creation (bool, optional): Whether this is a new project creation. Defaults to False.
            creation_position (int, optional): The position of the new project. Defaults to -1.

        Returns:
            project_dict (dict): A dictionary representation of the Project object.
        """
        from autogpt.projects.projects_broker import ProjectsBroker
        project_broker = ProjectsBroker()
        project_list = project_broker.get_projects()


        if( 0 <= project_position_number) :
            sub_folder_name = project_broker.project_dir_name_formater(project_name = self.project_name)
            for i , project in enumerate(project_list) :
                current_project_foldername = project_broker.project_dir_name_formater(project.project_name) 
                if (i == project_position_number and current_project_foldername != sub_folder_name ) :
                    # NOT_DELETING THE PROJECT BUT CREATING .backup
                    backup = f"{current_project_foldername}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.backup"
                    shutil.copy(current_project_foldername ,backup )
           
                elif i != project_position_number  and current_project_foldername== sub_folder_name :
                    # NOT_DELETING THE PROJECT BUT ADDING EXTENTION
                    sub_folder_name = sub_folder_name + '_1'
                    
            project_broker.create_project_dir(project_name = sub_folder_name)
            

        # lead_agent_dict = self.lead_agent.save()
        # delegated_agents_list = [agent.save() for agent in self.delegated_agents_list]
        
        # project_dict = {
        #     "project_name": self.project_name,
        #     "project_budget": self.project_budget,
        #     "lead_agent": lead_agent_dict,
        #     "delegated_agents": delegated_agents_list
        # }

        new_path = os.path.join(PROJECT_DIR, sub_folder_name , 'settings.yaml')

        with open(new_path, "w", encoding="utf-8") as file:
            yaml.dump(self.to_dict(), file, allow_unicode=True)
        
        return self.to_dict()
    

    def get_lead(self) -> AgentModel:
        """
        Returns the lead agent of the project.

        Returns:
            lead_agent (AgentModel): The lead agent of the project.
        """
        return  self.lead_agent
    
    def get_delegated_agents_list(self) -> list[AgentModel] : 
        """
        Returns the list of delegated agents for the project.

        Returns:
            delegated_agents_list (List[AgentModel]): The list of delegated agents for the project.
        """
        return  self.delegated_agents_list
    
    def delete_delegated_agents(self, position = int) -> bool: 
        """
        Deletes the delegated agent at the given position and returns True if successful.

        Args:
            position (int): The position of the delegated agent to be deleted.

        Returns:
            success (bool): True if the delegated agent was successfully deleted, False otherwise.
        """
        if 0 <= position <= len(self.delegated_agents_list) :
            return  False
        else : 
            self.delegated_agents_list(position) # Todo implement delete an agent
            
            #return True



        
