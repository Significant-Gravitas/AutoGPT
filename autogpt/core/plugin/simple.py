from importlib import import_module

from autogpt.core.plugin.base import (
    PluginLocation,
    PluginService,
    PluginStorageFormat,
    PluginStorageRoute,
    PluginType,
)


class SimplePluginService(PluginService):
    @staticmethod
    def get_plugin(plugin_location: dict | PluginLocation) -> PluginType:
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
                f"Plugin storage format {plugin_location.storage_format} is not implemented."
            )

    ####################################
    # Low-level storage format loaders #
    ####################################
    @staticmethod
    def load_from_file_path(plugin_route: PluginStorageRoute) -> PluginType:
        """Load a plugin from a file path."""
        # TODO: Define an on disk storage format and implement this.
        #   Can pull from existing zip file loading implementation
        raise NotImplemented("Loading from file path is not implemented.")

    @staticmethod
    def load_from_import_path(plugin_route: PluginStorageRoute) -> PluginType:
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
        raise NotImplemented("Resolving plugin name to path is not implemented.")

    #####################################
    # High-level storage format loaders #
    #####################################

    @staticmethod
    def load_from_workspace(plugin_route: PluginStorageRoute) -> PluginType:
        """Load a plugin from the workspace."""
        try:
            plugin = SimplePluginService.load_from_file_path(plugin_route)
        except NotImplemented:
            raise
        except Exception as e:
            raise NotImplemented(
                "Could not load plugin from workspace as a file path and "
                "no other loaders are implemented."
            )

        return plugin

    @staticmethod
    def load_from_installed_package(plugin_route: PluginStorageRoute) -> PluginType:
        try:
            plugin = SimplePluginService.load_from_import_path(plugin_route)
        except Exception as e:
            raise NotImplemented(
                "Could not load plugin from installed package as an import path and "
                "no other loaders are implemented."
            )
        return plugin
