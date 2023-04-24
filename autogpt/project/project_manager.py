import click
from pathlib import Path
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
        path = Path(f"projects/{project_name}/agents")
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

    def project_agents(project_name: str) -> None:
        """
        A command-line program that allows selecting an agent YAML file from a list of files in the specified project.

        Args:
            project_name (str): The name of the project.
        """
        project_manager = ProjectManager()
        agent_files = project_manager.get_agents(project_name)

        if not agent_files:
            click.echo("Not a valid project or no agents found.")
            return

        click.echo("Available AI-Assistants:")
        click.echo(f"Available {project_name} AI-Assistants:")
        agent_files_sorted = sorted(agent_files, key=lambda file: project_manager.read_ai_name(file))
        for idx, file in enumerate(agent_files_sorted, start=1):
            ai_name = project_manager.read_ai_name(file)
            click.echo(f"{idx}. {ai_name} (Config: {file.name})")

        choice = click.prompt("Select an AI-Assistant by entering the corresponding number", type=int)

        if 1 <= choice <= len(agent_files_sorted):
            selected_file = agent_files_sorted[choice - 1]
            selected_ai_name = project_manager.read_ai_name(selected_file)
            click.echo(f"{selected_ai_name} is now getting to work.")
            return selected_file
        else:
            click.echo("Invalid choice.")
            project_manager.project_agents(project_name)
