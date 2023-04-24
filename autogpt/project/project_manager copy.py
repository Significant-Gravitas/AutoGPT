import os
import shutil
from typing import Dict, List, Union
from autogpt.agent.agent_manager import AgentManager


class ProjectManager:
    def __init__(self, project_folder: str = "projects"):
        self.project_folder = project_folder
        if not os.path.exists(self.project_folder):
            os.makedirs(self.project_folder)
        self.agents: Dict[str, AgentManager] = {}

    def create_project(self, project_name: str) -> str:
        """
        Create a new project with the given name.

        Args:
            project_name (str): The name of the new project.

        Returns:
            str: A message indicating the result of the operation.
        """
        project_path = os.path.join(self.project_folder, project_name)
        if not os.path.exists(project_path):
            os.makedirs(project_path)
            self.agents[project_name] = AgentManager()
            return f"Project '{project_name}' has been created."
        else:
            return f"Project '{project_name}' already exists."

    def start_project(self, project_name: str) -> str:
        """
        Start working on the specified project.

        Args:
            project_name (str): The name of the project to start.

        Returns:
            str: A message indicating the result of the operation.
        """
        project_path = os.path.join(self.project_folder, project_name)
        if os.path.exists(project_path):
            return f"Working on project '{project_name}'."
        else:
            return f"Project '{project_name}' not found."

    def delete_project(self, project_name: str) -> str:
        """
        Delete the specified project.

        Args:
            project_name (str): The name of the project to delete.

        Returns:
            str: A message indicating the result of the operation.
        """
        project_path = os.path.join(self.project_folder, project_name)
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
            del self.agents[project_name]
            return f"Project '{project_name}' has been deleted."
        else:
            return f"Project '{project_name}' not found."

    def list_projects(self) -> List[str]:
        """
        List all projects.

        Returns:
            List[str]: A list of all project names.
        """
        return os.listdir(self.project_folder)

    def add_agent(self, project_name: str, agent_name: str) -> str:
        """
        Add an agent to the specified project.

        Args:
            project_name (str): The name of the project to add the agent to.
            agent_name (str): The name of the agent to add.

        Returns:
            str: A message indicating the result of the operation.
        """
        if project_name in self.agents:
            self.agents[project_name].create_agent(agent_name)
            return f"Agent '{agent_name}' added to project '{project_name}'."
        else:
            return f"Project '{project_name}' not found."
    
    def list_agents(self, project_name: str) -> Union[str, List[str]]:
        """
        List all agents in the specified project.

        Args:
            project_name (str): The name of the project to list agents for.

        Returns:
            Union[str, List[str]]: A list of agent names or an error message if the project is not found.
        """
        if project_name in self.agents:
            return [agent.name for agent in self.agents[project_name].list_agents()]
        else:
            return f"Project '{project_name}' not found."

    def remove_agent(self, project_name: str, agent_name: str) -> str:
        """
        Remove an agent from the specified project.

        Args:
            project_name (str): The name of the project to remove the agent from.
            agent_name (str): The name of the agent to remove.

        Returns:
            str: A message indicating the result of the operation.
        """
        if project_name in self.agents:
            self.agents[project_name].delete_agent(agent_name)
            return f"Agent '{agent_name}' removed from project '{project_name}'."
        else:
            return f"Project '{project_name}' not found."

