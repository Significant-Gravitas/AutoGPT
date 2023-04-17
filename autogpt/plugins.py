"""Handles loading of plugins."""

import zipfile
from ast import Module
from pathlib import Path
from typing import List, Optional, Tuple
from zipimport import zipimporter


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


def scan_plugins(plugins_path: Path, debug: bool = False) -> List[Tuple[str, Path]]:
    """Scan the plugins directory for plugins.

    Args:
        plugins_path (Path): Path to the plugins directory.

    Returns:
        List[Path]: List of plugins.
    """
    plugins = []
    for plugin in plugins_path.glob("*.zip"):
        if module := inspect_zip_for_module(str(plugin), debug):
            plugins.append((module, plugin))
    return plugins


def load_plugins(plugins_path: Path, debug: bool = False) -> List[Module]:
    """Load plugins from the plugins directory.

    Args:
        plugins_path (Path): Path to the plugins directory.

    Returns:
        List[Path]: List of plugins.
    """
    plugins = scan_plugins(plugins_path)
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
    return plugin_modules
