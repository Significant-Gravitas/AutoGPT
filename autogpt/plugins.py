"""Handles loading of plugins."""
import os
import zipfile
from glob import glob
from pathlib import Path
from zipimport import zipimporter
from typing import List, Optional, Tuple

from abstract_singleton import AbstractSingleton

from autogpt.config import Config


def inspect_zip_for_module(zip_path: str, debug: bool = False) -> Optional[str]:
    """
    Inspect a zipfile for a module.

    Args:
        zip_path (str): Path to the zipfile.
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        Optional[str]: The name of the module if found, else None.
    """
    with zipfile.ZipFile(zip_path, "r") as zfile:
        for name in zfile.namelist():
            if name.endswith("__init__.py"):
                if debug:
                    print(f"Found module '{name}' in the zipfile at: {name}")
                return name
    if debug:
        print(f"Module '__init__.py' not found in the zipfile @ {zip_path}.")
    return None


def scan_plugins(plugins_path: str, debug: bool = False) -> List[Tuple[str, Path]]:
    """Scan the plugins directory for plugins.

    Args:
        plugins_path (str): Path to the plugins directory.
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        List[Tuple[str, Path]]: List of plugins.
    """
    plugins = []
    plugins_path_path = Path(plugins_path)

    for plugin in plugins_path_path.glob("*.zip"):
        if module := inspect_zip_for_module(str(plugin), debug):
            plugins.append((module, plugin))
    return plugins


def blacklist_whitelist_check(plugins: List[AbstractSingleton], cfg: Config):
    """Check if the plugin is in the whitelist or blacklist.

    Args:
        plugins (List[Tuple[str, Path]]): List of plugins.
        cfg (Config): Config object.

    Returns:
        List[Tuple[str, Path]]: List of plugins.
    """
    loaded_plugins = []
    for plugin in plugins:
        if plugin.__name__ in cfg.plugins_blacklist:
            continue
        if plugin.__name__ in cfg.plugins_whitelist:
            loaded_plugins.append(plugin())
        else:
            ack = input(
                f"WARNNG Plugin {plugin.__name__} found. But not in the"
                " whitelist... Load? (y/n): "
            )
            if ack.lower() == "y":
                loaded_plugins.append(plugin())

    if loaded_plugins:
        print(f"\nPlugins found: {len(loaded_plugins)}\n" "--------------------")
    for plugin in loaded_plugins:
        print(f"{plugin._name}: {plugin._version} - {plugin._description}")
    return loaded_plugins


def load_plugins(cfg: Config = Config(), debug: bool = False) -> List[object]:
    """Load plugins from the plugins directory.

    Args:
        cfg (Config): Config instance inluding plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
        List[AbstractSingleton]: List of plugins initialized.
    """
    plugins = scan_plugins(cfg.plugins_dir)
    plugin_modules = []
    for module, plugin in plugins:
        plugin = Path(plugin)
        module = Path(module)
        if debug:
            print(f"Plugin: {plugin} Module: {module}")
        zipped_package = zipimporter(plugin)
        zipped_module = zipped_package.load_module(str(module.parent))
        for key in dir(zipped_module):
            if key.startswith("__"):
                continue
            a_module = getattr(zipped_module, key)
            a_keys = dir(a_module)
            if "_abc_impl" in a_keys and a_module.__name__ != "AutoGPTPluginTemplate":
                plugin_modules.append(a_module)
    loaded_plugin_modules = blacklist_whitelist_check(plugin_modules, cfg)
    return loaded_plugin_modules
