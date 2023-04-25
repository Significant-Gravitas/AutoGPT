# sourcery skip: do-not-use-staticmethod
"""
A module that contains the AIConfig class object that contains the configuration
"""
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional, Type, List
import shutil

import distro
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
            })
        # TODO __version (str): Version of the app
    """
        
    def __init__(self, project_number : int = -1,config_file: str = SAVE_FILE) -> None:
        """
        Initialize a class instance
        Parameters:
            project_number(int): The config entry number (0 to 4) to be updated or added.
            config_file(str): The path to the config yaml file.
              DEFAULT: "../agent_settings.yaml"
        Returns:
            None
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

    @classmethod
    def _load(cls, config_file: str = SAVE_FILE) -> list:
        """
        Loads the projects from the specified YAML file and returns a list of dictionaries containing the project parameters.

        Parameters:
            config_file(str): The path to the config yaml file.
            DEFAULT: "../agent_settings.yaml"
        
        Returns:
            agent_configs (list): A list of dictionaries containing the project parameters for each entry.
        """
        
        if not os.path.exists(config_file):
            cls._projects = []
            return cls._projects

        with open(config_file, encoding="utf-8") as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader)

        cls._projects = []
        version =  config_params.get("projects", '')
        if version != '' :
            version = version # Not supported for the moment
        for project in config_params.get("projects", []):
            project_name = project["project_name"]

            lead_agent_data = project["lead_agent"]
            lead_agent = AgentConfig(
                agent_name=lead_agent_data["agent_name"],
                agent_role=lead_agent_data["agent_role"],
                agent_goals=lead_agent_data["agent_goals"],
                agent_model=lead_agent_data.get("agent_model", None),
                
            )

            if (project["delegated_agents"]) :
                delegated_agents_list = []
                for delegated_agents_data in project["delegated_agents"]:
                    delegated_agents = AgentConfig(
                        agent_name=delegated_agents_data["agent_name"],
                        agent_role=delegated_agents_data["agent_role"],
                        agent_goals=delegated_agents_data["agent_goals"],
                        agent_model=delegated_agents_data.get("agent_model", None),
                        agent_model_type=delegated_agents_data.get("agent_model_type", None),
                    )
                    delegated_agents_list.append(delegated_agents)

            cls._projects.append(Project(project_name =  project_name,lead_agent = lead_agent, delegated_agents = delegated_agents_list))
            
            break 
            """
            # 
            This break allows to push the new YAML back-end
            A PR with thisbreak will not allow multiple projects but set the new architecture for multiple model
            """

        return cls._projects

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
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config_params = {}

        if "projects" not in config_params:
            config_params["projects"] = []

        if 0 <= self._current_project_id < len(config_params["projects"]):
            config_params["projects"][self._current_project_id] = self.get_current_project()
        else:
            config_params["projects"].append(self.get_current_project())

        data_to_save = {"version": "X.Y.X", "projects": config_params}
        with open(config_file, "w", encoding="utf-8") as file:
            yaml.dump(data_to_save, file, allow_unicode=True)

    def _save(self, config_file: str = SAVE_FILE) -> None:
        """
        Saves the current projecturation to the specified file yaml file path as a yaml file.

        Parameters:
            config_file(str): The path to the config yaml file.
            DEFAULT: "../agent_settings.yaml"
        Returns:
            None
        """
        project_data = []
        for project in self._projects:
            delegated_agents_data = []
            for delegated_agents in project["delegated_agents"]:
                delegated_agents_data.append(
                    {
                        "agent_name": delegated_agents.agent_name,
                        "agent_role": delegated_agents.agent_role,
                        "agent_goals": delegated_agents.agent_goals,
                        "agent_model": delegated_agents.agent_model,
                        "agent_model_type": delegated_agents.agent_model_type,
                    }
                )
            project_data.append(
                {
                    "project_name": project["project_name"],
                    "lead_agent": {
                        "agent_name": project["lead_agent"].agent_name,
                        "agent_role": project["lead_agent"].agent_role,
                        "agent_goals": project["lead_agent"].agent_goals,
                        "agent_model": project["lead_agent"].agent_model,
                        "agent_model_type": project["lead_agent"].agent_model_type,
                    },
                    "delegated_agents": delegated_agents_data,
                }
            )

        data_to_save = {"version": "X.Y.X", "projects": project_data}
        with open(config_file, "w", encoding="utf-8") as file:
            yaml.dump(data_to_save, file, allow_unicode=True)

    def construct_full_prompt(
        self, prompt_generator: Optional[PromptGenerator] = None
    ) -> str:
        """
        Returns a prompt to the user with the class information in an organized fashion.

        Parameters:
            None

        Returns:
            full_prompt (str): A string containing the initial prompt for the user
                including the agent_name, agent_role and agent_goals.
        """
        if self._current_project_id is None or self._current_project_id >= len(self._projects):
            raise ValueError("No project is currently selected.")

        current_project = self._projects[self._current_project_id]

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

        #prompt_generator.project_name = self._projects[self._current_project_id].project_name

        prompt_generator.goals = current_project.lead_agent.agent_goals
        prompt_generator.name = current_project.lead_agent.agent_name
        prompt_generator.role = current_project.lead_agent.agent_role
        prompt_generator.command_registry = current_project.lead_agent.command_registry
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
        for i, goal in enumerate(self._projects[self._current_project_id].lead_agent.agent_goals):
            full_prompt += f"{i+1}. {goal}\n"
        self._projects[self._current_project_id].lead_agent.prompt_generator = prompt_generator
        full_prompt += f"\n\n{prompt_generator.generate_prompt_string()}"
        return full_prompt

    def create_project(self, 
                    project_number : int, 
                    agent_name : str, 
                    agent_role : str, 
                    agent_goals : list, 
                    prompt_generator : str, 
                    command_registry : str ,
                    overwrite : bool = False) -> bool:
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
        number_of_config = len(self._projects)
        if project_number is None and number_of_config < MAX_AI_CONFIG:
            project_number = number_of_config

        if project_number > MAX_AI_CONFIG:
            raise ValueError(f"set_config: Value {project_number} not expected")

        # Check for duplicate config names
        for idx, config in enumerate(self._projects):
            if config["agent_name"] == agent_name and idx != project_number:
                print(f"Config with the name '{agent_name}' already exists")
                return False

        config = AgentConfig(
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            prompt_generator=prompt_generator,
            command_registry=command_registry)
        if project_number >= number_of_config:
            self._projects.append(config)
        else:
            self._projects[project_number].project_name = config

        self.set_project_number(new_project_number=project_number)
        self._save(project_number)

        return True
           
    def set_project_number(self, new_project_number: int) -> bool:
        """
        Sets the current projecturation number.

        Parameters:
            new_project_number (int): The new project number to be set.

        Returns:
            None
        """
        if new_project_number < 0 or new_project_number >= len(self._projects):
            raise ValueError(f"set_project_number: Value must be between 0 and {len(self._projects)-1}")
        self._current_project_id = new_project_number
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
    def __init__(self, project_name : str, lead_agent : AgentConfig, delegated_agents : List[AgentConfig] = []):
        self.project_name = project_name
        self.lead_agent = lead_agent
        self.delegated_agents = delegated_agents
        
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
