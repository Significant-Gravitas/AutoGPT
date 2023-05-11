"""The plugin system allows the Agent to be extended with new functionality."""
from autogpt.core.plugin.base import Plugin, PluginService
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.BASIC_DONE,
    handoff_notes=(
        "Before times: The plugin system has not been started yet.\n"
        "5/6-5/7: First draft of plugin interface complete.\n"
        "5/8: Plugin interface has been adjusted for use with the agent factory as we integrate with hello world. Still working on it.\n"
        "5/10: Interface is pretty solid at this point and limited version of initial implementation is done. Interface revisions submitted in PR and merged.\n"
        "5/11: Talked with Nick about plans for the plugin service including which plugins we'll support out of the box and which loading formats\n"
        "      we'll support.  Both are documented in the code. Also discussed how to handle listener plugins (punt, app server will deal with it).\n"
        "      and plugin metrics (punt, app server will deal with it).\n"
    ),
)
