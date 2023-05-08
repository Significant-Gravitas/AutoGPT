"""The plugin system allows the Agent to be extended with new functionality."""
from autogpt.core.plugin.base import Plugin, PluginManager

import autogpt.core.status

status = autogpt.core.status.Status.TODO
handover_notes = "The plugin system has not been started yet."
