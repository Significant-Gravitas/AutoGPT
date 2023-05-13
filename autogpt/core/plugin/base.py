import abc
import dataclasses
from typing import Type

from autogpt.core.budget import BudgetManager
from autogpt.core.command import Command, CommandRegistry
from autogpt.core.credentials import CredentialsManager
from autogpt.core.memory import MemoryBackend
from autogpt.core.model import EmbeddingModel, LanguageModel
from autogpt.core.planning import Planner

# Expand to other types as needed
PluginType = (
    Type[BudgetManager]  # Swappable maybe never (think about budgets for resources)
    | Type[Command]  # Swappable now
    | Type[CommandRegistry]  # Swappable maybe never
    | Type[CredentialsManager]  # Swappable never
    | Type[EmbeddingModel]  # Swappable soon
    | Type[LanguageModel]  # Swappable soon
    | Type[MemoryBackend]  # Swappable now
    | Type[Planner]  # Swappable soon
)

Plugin = (
    BudgetManager
    | Command
    | CommandRegistry
    | CredentialsManager
    | EmbeddingModel
    | LanguageModel
    | MemoryBackend
    | Planner
)


class PluginStorageFormat(str):
    """Supported plugin storage formats.

    Plugins can be stored at one of these supported locations.

    """

    INSTALLED_PACKAGE = "installed_package"  # Required now, loads system defaults
    WORKSPACE = "workspace"  # Required now
    # OPENAPI_URL = "open_api_url"           # Soon (requires some tooling we don't have yet).
    # OTHER_FILE_PATH = "other_file_path"    # Maybe later (maybe now)
    # GIT = "git"                            # Maybe later (or soon)
    # PYPI = "pypi"                          # Maybe later
    # AUTOGPT_PLUGIN_SERVICE = "autogpt_plugin_service"  # Long term solution, requires design
    # AUTO = "auto"                          # Feature for later maybe, automatically find plugin.


# Installed package example
# PluginLocation(
#     storage_format='installed_package',
#     storage_route='autogpt_plugins.twitter.SendTwitterMessage'
# )
# Workspace example
# PluginLocation(
#     storage_format='workspace',
#     storage_route='relative/path/to/plugin.pkl'
#     OR
#     storage_route='relative/path/to/plugin.py'
# )
# Git
# PluginLocation(
#     storage_format='git',
#     Exact format TBD.
#     storage_route='https://github.com/gravelBridge/AutoGPT-WolframAlpha/blob/main/autogpt-wolframalpha/wolfram_alpha.py'
# )
# PyPI
# PluginLocation(
#     storage_format='pypi',
#     storage_route='package_name'
# )


# PluginLocation(
#     storage_format='installed_package',
#     storage_route='autogpt_plugins.twitter.SendTwitterMessage'
# )


# A plugin storage route.
#
# This is a string that specifies where to load a plugin from
# (e.g. an import path or file path).
PluginStorageRoute = str


@dataclasses.dataclass
class PluginLocation:
    """A plugin location.

    This is a combination of a plugin storage format and a plugin storage route.
    It is used by the PluginService to load plugins.

    """

    storage_format: PluginStorageFormat
    storage_route: PluginStorageRoute


@dataclasses.dataclass
class PluginMetadata:
    """Metadata about a plugin."""

    name: str
    description: str
    type: PluginType
    location: PluginLocation


class PluginService(abc.ABC):
    """Base class for plugin service.

    The plugin service should be stateless. This defines the interface for
    loading plugins from various storage formats.

    """

    @staticmethod
    @abc.abstractmethod
    def get_plugin(plugin_location: PluginLocation) -> PluginType:
        """Get a plugin from a plugin location."""
        ...

    ####################################
    # Low-level storage format loaders #
    ####################################
    @staticmethod
    @abc.abstractmethod
    def load_from_file_path(plugin_route: PluginStorageRoute) -> PluginType:
        """Load a plugin from a file path."""

        ...

    @staticmethod
    @abc.abstractmethod
    def load_from_import_path(plugin_route: PluginStorageRoute) -> PluginType:
        """Load a plugin from an import path."""
        ...

    @staticmethod
    @abc.abstractmethod
    def resolve_name_to_path(
        plugin_route: PluginStorageRoute, path_type: str
    ) -> PluginStorageRoute:
        """Resolve a plugin name to a plugin path."""
        ...

    #####################################
    # High-level storage format loaders #
    #####################################

    @staticmethod
    @abc.abstractmethod
    def load_from_workspace(plugin_route: PluginStorageRoute) -> PluginType:
        """Load a plugin from the workspace."""
        ...

    @staticmethod
    @abc.abstractmethod
    def load_from_installed_package(plugin_route: PluginStorageRoute) -> PluginType:
        """Load a plugin from an installed package."""
        ...
