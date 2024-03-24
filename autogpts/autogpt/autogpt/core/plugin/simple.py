from importlib import import_module
from typing import TYPE_CHECKING

from autogpt.core.plugin.base import (
    PluginLocation,
    PluginService,
    PluginStorageFormat,
    PluginStorageRoute,
)

if TYPE_CHECKING:
    from autogpt.core.plugin.base import PluginType


class SimplePluginService(PluginService):
    @staticmethod
    def get_plugin(plugin_location: dict | PluginLocation) -> "PluginType":
        """Get a plugin from a plugin location."""
        if isinstance(plugin_location, dict):
            plugin_location = PluginLocation.parse_obj(plugin_location)
        if plugin_location.storage_format == PluginStorageFormat.WORKSPACE:
            return SimplePluginService.load_from_workspace(
                plugin_location.storage_route
            )
        elif plugin_location.storage_format == PluginStorageFormat.INSTALLED_PACKAGE:
            return SimplePluginService.load_from_installed_package(
                plugin_location.storage_route
            )
        else:
            raise NotImplementedError(
                "Plugin storage format %s is not implemented."
                % plugin_location.storage_format
            )

    ####################################
    # Low-level storage format loaders #
    ####################################
    @staticmethod
    def load_from_file_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from a file path."""
        # TODO: Define an on disk storage format and implement this.
        #   Can pull from existing zip file loading implementation
        raise NotImplementedError("Loading from file path is not implemented.")

    @staticmethod
    def load_from_import_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from an import path."""
        module_path, _, class_name = plugin_route.rpartition(".")
        return getattr(import_module(module_path), class_name)

    @staticmethod
    def resolve_name_to_path(
        plugin_route: PluginStorageRoute, path_type: str
    ) -> PluginStorageRoute:
        """Resolve a plugin name to a plugin path."""
        # TODO: Implement a discovery system for finding plugins by name from known
        #   storage locations. E.g. if we know that path_type is a file path, we can
        #   search the workspace for it. If it's an import path, we can check the core
        #   system and the auto_gpt_plugins package.
        raise NotImplementedError("Resolving plugin name to path is not implemented.")

    #####################################
    # High-level storage format loaders #
    #####################################

    @staticmethod
    def load_from_workspace(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from the workspace."""
        plugin = SimplePluginService.load_from_file_path(plugin_route)
        return plugin

    @staticmethod
    def load_from_installed_package(plugin_route: PluginStorageRoute) -> "PluginType":
        plugin = SimplePluginService.load_from_import_path(plugin_route)
        return plugin
