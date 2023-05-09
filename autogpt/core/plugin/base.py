import abc
import dataclasses
from enum import StrEnum
from typing import Type

from autogpt.core.budget import BudgetManager
from autogpt.core.command import Command, CommandRegistry
from autogpt.core.llm import LanguageModel
from autogpt.core.memory import MemoryBackend
from autogpt.core.planning import Planner

# Expand to other types as needed
PluginType = (
    Type[BudgetManager]
    | Type[Command]
    | Type[CommandRegistry]
    | Type[LanguageModel]
    | Type[MemoryBackend]
    | Type[Planner]
)

Plugin = (
    BudgetManager | Command | CommandRegistry | LanguageModel | MemoryBackend | Planner
)


class PluginStorageFormat(StrEnum):
    """Supported plugin storage formats."""

    AUTO = "auto"  # We'll try to determine the plugin load location
    AUTOGPT_PLUGIN_REPO = "autogpt_plugin_repo"  # Grab them from our managed repo
    WORKSPACE = "workspace"  # Grab them from the workspace
    OTHER_FILE_PATH = "other_file_path"  # Grab them from a file path
    INSTALLED_PACKAGE = "installed_package"  # Grab them from an installed package
    # PYPI = "pypi"  # Grab them from pypi
    # GITHUB = "github"  # Grab them from a github repo
    # AUTOGPT_PLUGIN_SERVICE = "autogpt_plugin_service"  # Grab them from a service


PluginStorageRoute = str


class PluginStorageRouteFormat(StrEnum):
    """Supported plugin storage route formats."""

    RAW_NAME = "raw_name"
    IMPORT_PATH = "import_path"
    FILE_PATH = "file_path"


PLUGIN_FORMAT_MAP = {
    PluginStorageFormat.AUTO: None,
    PluginStorageFormat.AUTOGPT_PLUGIN_REPO: (
        PluginStorageRouteFormat.RAW_NAME,
        PluginStorageRouteFormat.IMPORT_PATH,
    ),
    PluginStorageFormat.WORKSPACE: (
        PluginStorageRouteFormat.RAW_NAME,
        PluginStorageRouteFormat.FILE_PATH,
    ),
    PluginStorageFormat.OTHER_FILE_PATH: (
        PluginStorageRouteFormat.RAW_NAME,
        PluginStorageRouteFormat.FILE_PATH,
    ),
    PluginStorageFormat.INSTALLED_PACKAGE: (PluginStorageRouteFormat.IMPORT_PATH,),
}


@dataclasses.dataclass
class PluginLocation:
    """A plugin location."""

    storage_format: PluginStorageFormat
    storage_route: PluginStorageRoute


@dataclasses.dataclass
class PluginMetadata:
    name: str
    description: str
    type: PluginType
    location: PluginLocation


class PluginService(abc.ABC):
    """Base class for plugin service.

    The plugin service should be stateless. This defines
    """

    @staticmethod
    @abc.abstractmethod
    def get_plugin(plugin_location: PluginLocation) -> Plugin:
        """Get a plugin from a plugin location."""
        ...
