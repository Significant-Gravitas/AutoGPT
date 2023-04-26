import click
from colorama import Fore
from pathlib import Path
from autogpt.logs import logger
import yaml

class ProjectManager:
    def __init__(self, project_folder: str = "projects"):
        """
        Initialize a ProjectManager instance.

        Args:
            project_folder (str, optional): The project folder path. Defaults to "projects".
        """
        self.project_folder = project_folder
        if not Path.exists(Path(self.project_folder)):
            Path(self.project_folder).mkdir(parents=True)

    @staticmethod
    def get_agents(project_name: str) -> list:
        """
        Find YAML files in the specified project's agents folder.

        Args:
            project_name (str): The name of the project.

        Returns:
            list: A list of pathlib.Path objects representing YAML files found in the specified path.
        """
        path = Path(f"{self.project_folder}/{project_name}/agents")
        return list(path.glob("*.yaml"))

    @staticmethod
    def read_ai_name(yaml_file: Path) -> str:
        """
        Read the 'ai_name' value from the specified YAML file.

        Args:
            yaml_file (Path): A pathlib.Path object representing the YAML file.

        Returns:
            str: The 'ai_name' value from the YAML file.
        """
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data.get("ai_name")
        
    def get_project_config_path(self, project_name: str) -> Path:
        """
        Generate the project config file path based on the project name.

        Args:
            project_name (str): The name of the project.

        Returns:
            Path: The pathlib.Path object representing the project config file path.
        """
        project_config = f"project_{project_name.lower().replace(' ', '_')}.yaml"
        return Path(f"{self.project_folder}/{project_name}/{project_config}")

    def create_project_config(self, project_name: str, project_description: str) -> None:
        """
        Create a yaml file with the project details in the main project folder.

        Args:
            project_name (str): The name of the project.
            project_description (str): A brief description of the project.
            agents (list): A list of agent configs in the project.
        """
        # Ensure the folder structure exists
        project_folder_path = Path(f"{self.project_folder}/{project_name.lower().replace(' ', '_')}")
        project_folder_path.mkdir(parents=True, exist_ok=True)

        path = self.get_project_config_path(project_name)

        project_data = {
            "project_name": project_name,
            "project_description": project_description,
            "agents": []
        }

        with open(path, "w") as file:
            yaml.dump(project_data, file, allow_unicode=True)

    def load_project_config(self, project_name: str):
        """
        Load project details from the project config YAML file.

        Args:
            project_name (str): The name of the project.

        Returns:
            dict: A dictionary containing the project details.
        """
        path = self.get_project_config_path(project_name)

        if path.exists():
            with open(path, "r") as file:
                project_data = yaml.safe_load(file)
            return project_data
        else:
            raise FileNotFoundError(f"Project config file not found for project: {project_name}")

    def add_agent_to_project_config(self, project_name: str, agent: str) -> None:
        """
        Add a new agent to the project config YAML file.

        Args:
            project_name (str): The name of the project.
            agent_name (str): The name of the agent.
        """
        project_data = self.load_project_config(project_name)
        project_data["agents"].append(agent)

        path = self.get_project_config_path(project_name)

        with open(path, "w") as file:
            yaml.safe_dump(project_data, file)

    def project_agents(self, project_name: str) -> None:
        """
        A command-line program that allows selecting an agent YAML file from a list of files in the specified project.

        Args:
            project_name (str): The name of the project.
        """
        project_config = self.load_project_config(project_name)

        if not project_config or not project_config.get("agents"):
            logger.typewriter_log("Not a valid project or no agents found.", Fore.YELLOW)
            return

        project_full_name = project_config.get("project_name", project_name)
        logger.typewriter_log(f"Available {project_full_name} AI-Assistants:")

        agents = project_config["agents"]
        for idx, agent_name in enumerate(agents, start=1):
            logger.typewriter_log(f"[{idx}] ", Fore.GREEN, f"{agent_name}")

        choice = click.prompt("Select an AI-Assistant by entering the corresponding number", type=int)

        if 1 <= choice <= len(agents):
            selected_ai_name = agents[choice - 1]
            logger.typewriter_log(f"{selected_ai_name} is now getting to work.",)
            return selected_ai_name
        else:
            logger.typewriter_log("Invalid choice.", Fore.YELLOW)
            self.project_agents(project_name)
