import abc
import dataclasses
import typing
from enum import Enum

if typing.TYPE_CHECKING:
    from autogpt.core.command import Command
    from autogpt.core.configuration import Configuration
    from autogpt.core.logging import Logger
    from autogpt.core.workspace import Workspace


class PluginType(Enum):
    COMMAND = 0
    OPENAPI = 1
    SYSTEM_REPLACEMENT = 2
    MESSAGE_SINK = 3
    LOGGING_SINK = 4


@dataclasses.dataclass
class Plugin:
    name: str
    description: str
    type: PluginType
    # This will have other plugin types added here
    plugin_object: typing.Union[Command, None]


class PluginManager(abc.ABC):
    @abc.abstractproperty
    def default_configration():
        return {"plugin_load_location": ".", "plugins_to_load": "*"}

    @abc.abstractmethod
    def __init__(
        self, workspace: Workspace, configuration: Configuration, logger: Logger
    ) -> None:
        pass

    @abc.abstractmethod
    def gather_plugins(type: PluginType) -> typing.List[Plugin]:
        """Gathers and returns a list of plugins that can be used by other systems to register their plugins"""
        pass
