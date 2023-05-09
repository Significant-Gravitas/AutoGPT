"""The plugin system allows the Agent to be extended with new functionality."""
from autogpt.core.plugin.base import Plugin, PluginManager
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.TODO,
    handoff_notes="The plugin system has not been started yet.",
)
