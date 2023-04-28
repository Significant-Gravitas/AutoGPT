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

from typing import Optional, Type, List
from autogpt.projects.agent_model import AgentModel


class Project:
    """
    A class representing an AI project with a name, API budget, lead agent, and delegated agents.

    Attributes:
        project_name (str): The name of the project.
        api_budget (float): The budget allocated for using the OpenAI API.
        lead_agent (AgentConfig): The lead agent for the project.
        delegated_agents (List[AgentConfig]): A list of delegated agents for the project.

    Methods:
        __init__(self, project_name: str, api_budget: float, lead_agent: AgentConfig, delegated_agents: List[AgentConfig] = [])
            Initializes a new Project object with the given parameters.
        __str__(self) -> str
            Returns a string representation of the Project object.
        __toDict__(self) -> dict
            Returns a dictionary representation of the Project object.
    """
    def __init__(self, 
                 project_name : str, 
                 api_budget : float , 
                 lead_agent : AgentModel, 
                 delegated_agents : List[AgentModel] = []):        
        """
        Initializes the Project class with the given attributes.

        Args:
            project_name (str): The name of the project.
            api_budget (float): The API budget for the project.
            lead_agent (AgentConfig): The lead agent configuration.
            delegated_agents (List[AgentConfig], optional): A list of delegated agent configurations. Defaults to an empty list.
        """
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
        """
        Converts the Project object to a dictionary representation.

        Returns:
            dict_representation (dict): A dictionary representation of the Project object.
        """
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

    @classmethod
    def load(cls, config_params: dict) -> "Project":
        """
        Loads a portion of the configuration parameters and returns a Project instance.

        Args:
            config_params (dict): A dictionary containing the configuration parameters for the project.

        Returns:
            project_instance (Project): A Project instance with the loaded configuration parameters.
        """
        version =  config_params.get("version", '')
        if version != '' :
            cls._version = version # Not supported for the moment

        if (config_params.get("project_name")) :
            project_name = config_params["project_name"]
        else :
            raise ValueError("No project_name in the project.")

        if (config_params.get("budget")) :
            api_budget = config_params["budget"]
        else :
            raise ValueError("No budget in the project.")

        if config_params.get("lead_agent"):
            lead_agent_data = config_params["lead_agent"]
            lead_agent = AgentModel.load(lead_agent_data)
        else:
            raise ValueError("No lead_agent in the project.")

        delegated_agents_list = []
        if config_params.get("delegated_agents"):
            for delegated_agent_data in config_params["delegated_agents"]:
                delegated_agent = AgentModel.load(delegated_agent_data)
                delegated_agents_list.append(delegated_agent)

        project_instance = cls(project_name, api_budget, lead_agent, delegated_agents_list)
        return project_instance
    
    def save(self) -> dict:
        """
        Saves the Project object as a dictionary representation.

        Returns:
            project_dict (dict): A dictionary representation of the Project object.
        """
        lead_agent_dict = self.lead_agent.save()
        delegated_agents_list = [agent.save() for agent in self.delegated_agents]
        
        project_dict = {
            "project_name": self.project_name,
            "api_budget": self.api_budget,
            "lead_agent": lead_agent_dict,
            "delegated_agents": delegated_agents_list
        }
        return project_dict


        
