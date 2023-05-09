"""The plugin system allows the Agent to be extended with new functionality."""
from autogpt.core.plugin.base import Plugin, PluginService
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.TODO,
    handoff_notes=(
        "Before times: The plugin system has not been started yet.\n"
        "5/6-5/7: First draft of plugin interface complete.\n"
        "5/8: Plugin interface has been adjusted for use with the agent factory as we integrate with hello world. Still working on it."
    ),
)
