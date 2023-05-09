from autogpt.core.plugin.base import Plugin, PluginLocation, PluginService


class SimplePluginService(PluginService):
    @staticmethod
    def get_plugin(plugin_location: PluginLocation) -> Plugin:
        """Get a plugin from a plugin location."""
        # TODO: Implement loading from each of the storage route formats.
        # TODO: Implement loading from each of the storage formats using the appropriate
        #   storage route format loaders.
        raise NotImplemented
