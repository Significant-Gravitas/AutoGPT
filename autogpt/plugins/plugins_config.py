from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

import yaml

if TYPE_CHECKING:
    from autogpt.config import Config

from pydantic import BaseModel

from autogpt.logs import logger
from autogpt.plugins.plugin_config import PluginConfig


class PluginsConfig(BaseModel):
    """Class for holding configuration of all plugins"""

    plugins: dict[str, PluginConfig]

    def __repr__(self):
        return f"PluginsConfig({self.plugins})"

    def get(self, name: str) -> Union[PluginConfig, None]:
        return self.plugins.get(name)

    def is_enabled(self, name) -> bool:
        plugin_config = self.plugins.get(name)
        return plugin_config is not None and plugin_config.enabled

    @classmethod
    def load_config(cls, global_config: Config) -> "PluginsConfig":
        empty_config = cls(plugins={})

        try:
            config_data = cls.deserialize_config_file(global_config=global_config)
            if type(config_data) != dict:
                logger.error(
                    f"Expected plugins config to be a dict, got {type(config_data)}, continuing without plugins"
                )
                return empty_config
            return cls(plugins=config_data)

        except BaseException as e:
            logger.error(
                f"Plugin config is invalid, continuing without plugins. Error: {e}"
            )
            return empty_config

    @classmethod
    def deserialize_config_file(cls, global_config: Config) -> dict[str, PluginConfig]:
        plugins_config_path = global_config.plugins_config_file
        if not os.path.exists(plugins_config_path):
            logger.warn("plugins_config.yaml does not exist, creating base config.")
            cls.create_empty_plugins_config(global_config=global_config)

        with open(plugins_config_path, "r") as f:
            plugins_config = yaml.load(f, Loader=yaml.FullLoader)

        plugins = {}
        for name, plugin in plugins_config.items():
            if type(plugin) == dict:
                plugins[name] = PluginConfig(
                    name=name,
                    enabled=plugin.get("enabled", False),
                    config=plugin.get("config", {}),
                )
            elif type(plugin) == PluginConfig:
                plugins[name] = plugin
            else:
                raise ValueError(f"Invalid plugin config data type: {type(plugin)}")
        return plugins

    @staticmethod
    def create_empty_plugins_config(global_config: Config):
        """Create an empty plugins_config.yaml file. Fill it with values from old env variables."""
        base_config = {}

        logger.debug(f"Legacy plugin denylist: {global_config.plugins_denylist}")
        logger.debug(f"Legacy plugin allowlist: {global_config.plugins_allowlist}")

        # Backwards-compatibility shim
        for plugin_name in global_config.plugins_denylist:
            base_config[plugin_name] = {"enabled": False, "config": {}}

        for plugin_name in global_config.plugins_allowlist:
            base_config[plugin_name] = {"enabled": True, "config": {}}

        logger.debug(f"Constructed base plugins config: {base_config}")

        logger.debug(f"Creating plugin config file {global_config.plugins_config_file}")
        with open(global_config.plugins_config_file, "w+") as f:
            f.write(yaml.dump(base_config))
            return base_config
