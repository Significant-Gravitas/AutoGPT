import uuid
from typing import Optional

from importlib import import_module
from pydantic import BaseModel, Field

from autogpt.core.configuration import SystemConfiguration, SystemSettings
from autogpt.core.memory.base import Memory
from autogpt.core.plugin.simple import PluginLocation
from autogpt.core.workspace.simple import SimpleWorkspace


class BaseAgentSystems(SystemConfiguration):
    memory :str  = "autogpt.core.memory.base.Memory"
    workspace : str = "autogpt.core.workspace.SimpleWorkspace"

    class Config(SystemConfiguration.Config):
        extra = "allow"

    @classmethod
    #def load_from_import_path(cls, attr) -> "Configurable":
    def load_from_import_path(cls, system_location : str):
        """Load a plugin from an import path."""
        module_path, _, class_name = system_location.rpartition(".")
        return getattr(import_module(module_path), class_name)

class BaseAgentConfiguration(SystemConfiguration):
    cycle_count : int =0
    max_task_cycle_count : int =3
    creation_time : str=""
    systems: BaseAgentSystems =BaseAgentSystems()
    user_id: Optional[uuid.UUID] = Field(default=None)
    agent_id: Optional[uuid.UUID] = Field(default=None)

    class Config(SystemConfiguration.Config):
        extra = "allow"


class BaseAgentSettings(BaseModel):
    #agent: BaseAgentSystemSettings 
    agent_class: str
    memory: Memory.SystemSettings = Memory.SystemSettings()
    workspace: SystemSettings = SimpleWorkspace.SystemSettings()

    class Config(BaseModel.Config):
        # This is a list of Field to Exclude during serialization
        json_encoders = {uuid.UUID: lambda v: str(v)}  # Custom encoder for UUID4
        extra = "allow"
        default_exclude = {
            "agent",
            "workspace",
            "planning",
            "chat_model_provider",
            "memory",
            "tool_registry",
        }

    def dict(self, remove_technical_values=True, *args, **kwargs):
        """
        Serialize the object to a dictionary representation.

        Args:
            remove_technical_values (bool, optional): Whether to exclude technical values. Default is True.
            *args: Additional positional arguments to pass to the base class's dict method.
            **kwargs: Additional keyword arguments to pass to the base class's dict method.
            kwargs['exclude'] excludes the fields from the serialization

        Returns:
            dict: A dictionary representation of the object.
        """
        self.prepare_values_before_serialization()  # Call the custom treatment before .dict()

        kwargs["exclude"] = kwargs.get("exclude", set())  # Get the exclude_arg
        if remove_technical_values:
            # Add the default technical fields to exclude_arg
            kwargs["exclude"] |= self.__class__.Config.default_exclude

        # Call the .dict() method with the updated exclude_arg
        return super().dict(*args, **kwargs)

    def json(self, remove_technical_values=True, *args, **kwargs):
        """
        Serialize the object to a dictionary representation.

        Args:
            remove_technical_values (bool, optional): Whether to exclude technical values. Default is True.
            *args: Additional positional arguments to pass to the base class's dict method.
            **kwargs: Additional keyword arguments to pass to the base class's dict method.
            kwargs['exclude'] excludes the fields from the serialization

        Returns:
            dict: A dictionary representation of the object.
        """
        self.prepare_values_before_serialization()  # Call the custom treatment before .json()

        kwargs["exclude"] = kwargs.get("exclude", set())  # Get the exclude_arg
        if remove_technical_values:
            # Add the default technical fields to exclude_arg
            kwargs["exclude"] |= self.__class__.Config.default_exclude

        return super().json(*args, **kwargs)

    # NOTE : To be implemented in the future
    def load_root_values(self, *args, **kwargs):
        pass  # NOTE : Currently not used
        self.agent_name = self.agent.configuration.agent_name
        self.agent_role = self.agent.configuration.agent_role
        self.agent_goals = self.agent.configuration.agent_goals
        self.agent_goal_sentence = self.agent.configuration.agent_goal_sentence

    # TODO Implement a BaseSettings class and move it to the BaseSettings ?
    def prepare_values_before_serialization(self):
        self.agent_setting_module = (
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self.agent_setting_class = self.__class__.__name__

    def update_agent_name_and_goals(self, agent_goals: dict) -> None:
        for key, value in agent_goals.items():
            # if key != 'agent' and key != 'workspace'  :
            setattr(self, key, value)


class BaseAgentDirectives(dict):
    """An object that contains the basic directives for the AI prompt.

    Attributes:
        constraints (list): A list of constraints that the AI should adhere to.
        resources (list): A list of resources that the AI can utilize.
        best_practices (list): A list of best practices that the AI should follow.
    """

    constraints: list[str]
    resources: list[str]
    best_practices: list[str]
