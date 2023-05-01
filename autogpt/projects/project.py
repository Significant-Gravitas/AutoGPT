"""
This module provides a class representing anproject with a name, API budget, lead agent, and delegated agents.
 
Classes:
- Project: A class representing anproject with a name, API budget, lead agent, and delegated agents.

Functions:
- None

Global Variables:
- None

Dependencies:
- typing: A built-in module for type hints and annotations.
- AgentConfig: A class representing the configuration settings for anagent.

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
import uuid

from autogpt.projects.agent_model import AgentModel
from pathlib import Path

SAVE_FILE = Path.cwd() / "ai_settings.yaml"
PROJECT_DIR = "autogpt/projects"
MAX_NB_PROJECT = 1
AUTOGPT_VERSION = 'X.Y.Z'  # TODO, implement in config.py or main or technical env file

class Project:
    """
    A class representing anproject with a name, API budget, lead agent, and delegated agents.

    Attributes:
        project_name (str): The name of the project.
        project_budget (float): The budget allocated for using the OpenAI API.
        lead_agent (AgentModel): The lead agent for the project.
        delegated_agents (List[AgentModel]): A list of delegated agents for the project.
        version (str): The version of the project.
        project_memory (Optional[str]): The memory used by the project.
        project_working_directory (Optional[str]): The working directory of the project.
        project_env (Optional[str]): The environment of the project.
        project_log_activity (Optional[str]): The log activity of the project.
        project_log_env (Optional[str]): The log environment of the project.

    Methods:
        __init__(self, project_name: str, project_budget: float, lead_agent: AgentModel,
                 delegated_agents: List[AgentModel] = [], version: str = "0.0.0",
                 project_memory: Optional[str] = None, project_working_directory: Optional[str] = None,
                 project_env: Optional[str] = None, project_log_activity: Optional[str] = None,
                 project_log_env: Optional[str] = None, team_name: Optional[str] = None) -> None
            Initializes a new Project object with the given parameters.
        load(cls, config_params: dict) -> "Project"
            Loads a portion of the configuration parameters and returns a Project instance.
        save(self, project_position_number : int, is_creation: bool = False  ) -> "Project"
            Saves the Project object as a dictionary representation.
        get_lead(self) -> AgentModel
            Returns the lead agent of the project.
        get_delegated_agents(self) -> List[AgentModel]
            Returns the list of delegated agents for the project.
        delete_delegated_agents(self, position: int) -> bool
            Deletes the delegated agent at the given position and returns True if successful.
        _check_method_load(cls, config_params: dict) -> bool
            Checks if the given configuration parameters are valid for loading a Project object.
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
            delegated_agents (List[AgentModel], optional): A list of delegated agents for the project. Defaults to an empty list.
            version (str, optional): The version of the project. Defaults to "0.0.0".
            project_memory (str, optional): The memory used by the project. Defaults to None.
            project_working_directory (str, optional): The working directory of the project. Defaults to None.
            project_env (str, optional): The environment of the project. Defaults to None.
            project_log_activity (str, optional): The log activity of the project. Defaults to None.
            project_log_env (str, optional): The log environment of the project. Defaults to None.
            team_name (str, optional): The name of the team. Defaults to None.
        """
        self.version = version
        self.uniq_id = str(uuid.uuid4())
        self.project_name = project_name
        self.project_budget = project_budget
        self.project_memory = project_memory
        self.project_working_directory = project_working_directory
        self.project_env = project_env
        self.project_log_activity = project_log_activity
        self.project_log_env = project_log_env

        team_name  = team_name or project_name
        self.agent_team = AgentTeam(team_name = team_name,
                                    lead_agent = lead_agent,
                                    delegated_agents = delegated_agents)

    #saving in Yaml
    def __str__(self) -> str:
        """
        Returns a string representation of the Project object.

        Returns:
            str_representation (str): A string representation of the Project object.
        """
        return str(self.to_dict())
    
    def to_dict(self) -> dict:
        """
        Converts the Project object to a dictionary representation.

        Returns:
            dict_representation (dict): A dictionary representation of the Project object.
        """
        dict_representation = {
            "project_name": self.project_name,
            "project_budget": self.project_budget,
            'agent_team' : self.agent_team.to_dict() 
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
             
        if cls._check_args_method_load( config_params) : 
            from autogpt.projects.projects_broker import ProjectsBroker 
            from autogpt.projects.agent_model import AgentModel  
            project_name = config_params["project_name"]
            project_budget = config_params["project_budget"]
            version =  config_params.get("version", ProjectsBroker.AUTOGPT_VERSION)
            project_memory = config_params.get("project_memory")
            project_working_directory = config_params.get("project_working_directory")
            project_env = config_params.get("project_env")
            project_log_activity = config_params.get("project_log_activity")
            project_log_env = config_params.get("project_log_env")

            agent_team = config_params["agent_team"]
            team_name = agent_team["team_name"]
            lead_agent_data = agent_team["lead_agent"]
            lead_agent = AgentModel.load_agent(lead_agent_data)
       
            delegated_agents = []
            if agent_team.get("delegated_agents"):
                for delegated_agent_data in agent_team["delegated_agents"]:
                    delegated_agent = AgentModel.load_agent(delegated_agent_data)
                    delegated_agents.append(delegated_agent)

        # Returns a Projects
        return cls(version = version, 
                   project_name = project_name, 
                   project_budget = project_budget,
                   project_memory = project_memory, 
                   project_working_directory = project_working_directory, 
                   project_env = project_env,
                   project_log_activity = project_log_activity, 
                   project_log_env = project_log_env, 
                   team_name = team_name, 
                   lead_agent = lead_agent, 
                   delegated_agents = delegated_agents)
    

    def save(self, project_position_number : int, is_creation : bool = False  ) -> "Project" :
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

        createdir = True 
        if( 0 <= project_position_number) :
            sub_folder_name = project_broker.project_dir_name_formater(project_name = self.project_name)
            for i , project in enumerate(project_list) :
                current_project_foldername = project_broker.project_dir_name_formater(project.project_name) 
                # NOT ERASING PROJECT THE PROJECT BUT CREATING .backup
                if (i == project_position_number and current_project_foldername != sub_folder_name ) :
                    backup = f"{current_project_foldername}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.backup"
                    shutil.copy(current_project_foldername ,backup )

                # IF A PROJECT WITH SAME NAME EXIST ADD EXTENTION          
                elif i != project_position_number  and current_project_foldername== sub_folder_name :
                    sub_folder_name = sub_folder_name + '_1'

                # DO NOT CREATE A REPOSITORY IF THE PROJECT ALREADY EXIST
                elif i == project_position_number and current_project_foldername != sub_folder_name :
                    createdir = True  

                # TODO : REMOVE THE 3 LINES IF YOU WORK ON PROJECT
                if i != project_position_number :
                    shutil.rmtree(current_project_foldername, ignore_errors = True)
                    createdir = True  
                        
            
            # lead_agent_dict = self.lead_agent.save()
            # delegated_agents = [agent.save() for agent in self.delegated_agents]
            
            # project_dict = {
            #     "project_name": self.project_name,
            #     "project_budget": self.project_budget,
            #     "lead_agent": lead_agent_dict,
            #     "delegated_agents": delegated_agents
            # }
            new_path = os.path.join(PROJECT_DIR, sub_folder_name , 'settings.yaml')

            project_dict = self.to_dict()

            # TODO only create a dir not error got raised
            if createdir == True : 
                project_broker.create_project_dir(project_name = sub_folder_name)
            with open(new_path, "w", encoding="utf-8") as file:
                yaml.dump(project_dict, file, allow_unicode=True)

            return self
    

    def get_lead(self) -> AgentModel:
        """
        Returns the lead agent of the project.

        Returns:
            lead_agent (AgentModel): The lead agent of the project.
        """
        return  self.agent_team.lead_agent
    
    def get_delegated_agents(self) -> list[AgentModel] : 
        """
        Returns the list of delegated agents for the project.

        Returns:
            delegated_agents (List[AgentModel]): The list of delegated agents for the project.
        """
        return  self.agent_team.delegated_agents
    
    def delete_delegated_agents(self, position = int) -> bool: 
        """
        Deletes the delegated agent at the given position and returns True if successful.

        Args:
            position (int): The position of the delegated agent to be deleted.

        Returns:
            success (bool): True if the delegated agent was successfully deleted, False otherwise.
        """
        if 0 <= position <= len(self.agent_team.delegated_agents) :
            return  False
        else : 
            self.agent_team.delegated_agents(position) # Todo implement delete an agent
            
            #return True

    @classmethod
    def _check_args_method_load(cls, config_params) -> bool :
        """
        Checks if the given configuration parameters are valid for loading a Project object.

        Args:
            config_params (dict): A dictionary containing the configuration parameters for the project.

        Returns:
            is_valid (bool): True if the configuration parameters are valid, False otherwise.
        """
        if  not config_params.get("project_name") :
            raise ValueError("Project.load() No project_name in the project.")

        if not isinstance(config_params.get("project_budget"), float):
            raise ValueError("Project.load() No budget in the project.")

        # Setting agent team
        agent_team = config_params.get("agent_team", None)
        team_name = agent_team.get("team_name", None)
        if not agent_team or not team_name :
            raise ValueError("Project.load() No lead_agent in the project.")
        
        return True
    
    # A class to generate UUID so plugin ay overid it 
    # https://github.com/Significant-Gravitas/Auto-GPT/discussions/3392#step-2-discussed-features
    def generate_uniqid(self) -> uuid : 
        return str(uuid.uuid4())
        



# NOTE : Keep it simple or set a long term structure (Pain in the ass)    
# NOTE : Not seing it as very useful    
class AgentTeam:
    """
    A class representing a team of agents for anproject.

    Attributes:
        team_name (str): The name of the team.
        lead_agent (AgentModel): The lead agent for the team.
        delegated_agents (List[AgentModel]): A list of delegated agents for the team.

    Methods:
        __init__(self, team_name: str, lead_agent: AgentModel, delegated_agents: List[AgentModel])
            Initializes a new AgentTeam object with the given parameters.
        to_dict(self) -> dict
            Converts the AgentTeam object to a dictionary representation.

    """
    def __init__(self, team_name, lead_agent, delegated_agents):
        """
        Initializes a new AgentTeam object with the given parameters.

        Args:
            team_name (str): The name of the team.
            lead_agent (AgentModel): The lead agent for the team.
            delegated_agents (List[AgentModel]): A list of delegated agents for the team.
        """
        
        self.team_name = team_name
        self.lead_agent = lead_agent
        self.delegated_agents = delegated_agents
        
        # CREATE A LIST OF AGENT IF EVER TO BE USED 
        # NOT PUBLICLY AVAILABLE AS DEEMED DANGEROUS
        self._all_agent = []
        self._all_agent.append(lead_agent)
        for agent in delegated_agents :
            self._all_agent.append(agent)

    def to_dict(self) -> dict : 
        """
        Converts the AgentTeam object to a dictionary representation.

        Returns:
            dict_representation (dict): A dictionary representation of the AgentTeam object.
        """
        lead_agent_dict = self.lead_agent.to_dict() 

        delegated_agents = []
        for agent in self.delegated_agents:
            agent_dict = agent.to_dict() 
            delegated_agents.append(agent_dict)

        return  {
            'team_name' : self.team_name,
            'lead_agent' : lead_agent_dict,
            'delegated_agents' : delegated_agents
        }
    
# TODO : test & migrate to this function
def object_to_dict(obj):
    obj_dict = {}
    for key in dir(obj):
        if not key.startswith('__'):
            value = getattr(obj, key)
            if not callable(value):
                if isinstance(value, object):
                    obj_dict[key] = object_to_dict(value)
                elif isinstance(value, list):
                    obj_dict[key] = []
                    for item in value:
                        if isinstance(item, object):
                            obj_dict[key].append(object_to_dict(item))
                        else:
                            obj_dict[key].append(item)
                else:
                    obj_dict[key] = value
    return obj_dict


    

