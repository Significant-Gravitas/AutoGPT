"""Handles loading of plugins."""
from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, List
from urllib.parse import urlparse
from zipimport import ZipImportError, zipimporter

import openapi_python_client
import requests
from auto_gpt_plugin_template import AutoGPTPluginTemplate
from openapi_python_client.config import Config as OpenAPIConfig

if TYPE_CHECKING:
    from autogpt.config import Config

from autogpt.models.base_open_ai_plugin import BaseOpenAIPlugin

logger = logging.getLogger(__name__)


def inspect_zip_for_modules(zip_path: str) -> list[str]:
    """
    Inspect a zipfile for a modules.

    Args:
        zip_path (str): Path to the zipfile.
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        list[str]: The list of module names found or empty list if none were found.
    """
    result = []
    with zipfile.ZipFile(zip_path, "r") as zfile:
        for name in zfile.namelist():
            if name.endswith("__init__.py") and not name.startswith("__MACOSX"):
                logger.debug(f"Found module '{name}' in the zipfile at: {name}")
                result.append(name)
    if len(result) == 0:
        logger.debug(f"Module '__init__.py' not found in the zipfile @ {zip_path}.")
    return result


def write_dict_to_json_file(data: dict, file_path: str) -> None:
    """
    Write a dictionary to a JSON file.
    Args:
        data (dict): Dictionary to write.
        file_path (str): Path to the file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def fetch_openai_plugins_manifest_and_spec(config: Config) -> dict:
    """
    Fetch the manifest for a list of OpenAI plugins.
        Args:
        urls (List): List of URLs to fetch.
    Returns:
        dict: per url dictionary of manifest and spec.
    """
    # TODO add directory scan
    manifests = {}
    for url in config.plugins_openai:
        openai_plugin_client_dir = f"{config.plugins_dir}/openai/{urlparse(url).netloc}"
        create_directory_if_not_exists(openai_plugin_client_dir)
        if not os.path.exists(f"{openai_plugin_client_dir}/ai-plugin.json"):
            try:
                response = requests.get(f"{url}/.well-known/ai-plugin.json")
                if response.status_code == 200:
                    manifest = response.json()
                    if manifest["schema_version"] != "v1":
                        logger.warning(
                            "Unsupported manifest version: "
                            f"{manifest['schem_version']} for {url}"
                        )
                        continue
                    if manifest["api"]["type"] != "openapi":
                        logger.warning(
                            f"Unsupported API type: {manifest['api']['type']} for {url}"
                        )
                        continue
                    write_dict_to_json_file(
                        manifest, f"{openai_plugin_client_dir}/ai-plugin.json"
                    )
                else:
                    logger.warning(
                        f"Failed to fetch manifest for {url}: {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error while requesting manifest from {url}: {e}")
        else:
            logger.info(f"Manifest for {url} already exists")
            manifest = json.load(open(f"{openai_plugin_client_dir}/ai-plugin.json"))
        if not os.path.exists(f"{openai_plugin_client_dir}/openapi.json"):
            openapi_spec = openapi_python_client._get_document(
                url=manifest["api"]["url"], path=None, timeout=5
            )
            write_dict_to_json_file(
                openapi_spec, f"{openai_plugin_client_dir}/openapi.json"
            )
        else:
            logger.info(f"OpenAPI spec for {url} already exists")
            openapi_spec = json.load(open(f"{openai_plugin_client_dir}/openapi.json"))
        manifests[url] = {"manifest": manifest, "openapi_spec": openapi_spec}
    return manifests


def create_directory_if_not_exists(directory_path: str) -> bool:
    """
    Create a directory if it does not exist.
    Args:
        directory_path (str): Path to the directory.
    Returns:
        bool: True if the directory was created, else False.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logger.debug(f"Created directory: {directory_path}")
            return True
        except OSError as e:
            logger.warning(f"Error creating directory {directory_path}: {e}")
            return False
    else:
        logger.info(f"Directory {directory_path} already exists")
        return True


def initialize_openai_plugins(manifests_specs: dict, config: Config) -> dict:
    """
    Initialize OpenAI plugins.
    Args:
        manifests_specs (dict): per url dictionary of manifest and spec.
        config (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
        dict: per url dictionary of manifest, spec and client.
    """
    openai_plugins_dir = f"{config.plugins_dir}/openai"
    if create_directory_if_not_exists(openai_plugins_dir):
        for url, manifest_spec in manifests_specs.items():
            openai_plugin_client_dir = f"{openai_plugins_dir}/{urlparse(url).hostname}"
            _meta_option = (openapi_python_client.MetaType.SETUP,)
            _config = OpenAPIConfig(
                **{
                    "project_name_override": "client",
                    "package_name_override": "client",
                }
            )
            prev_cwd = Path.cwd()
            os.chdir(openai_plugin_client_dir)

            if not os.path.exists("client"):
                client_results = openapi_python_client.create_new_client(
                    url=manifest_spec["manifest"]["api"]["url"],
                    path=None,
                    meta=_meta_option,
                    config=_config,
                )
                if client_results:
                    logger.warning(
                        f"Error creating OpenAPI client: {client_results[0].header} \n"
                        f" details: {client_results[0].detail}"
                    )
                    continue
            spec = importlib.util.spec_from_file_location(
                "client", "client/client/client.py"
            )
            module = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(module)
            finally:
                os.chdir(prev_cwd)

            client = module.Client(base_url=url)
            manifest_spec["client"] = client
    return manifests_specs


def instantiate_openai_plugin_clients(manifests_specs_clients: dict) -> dict:
    """
    Instantiates BaseOpenAIPlugin instances for each OpenAI plugin.
    Args:
        manifests_specs_clients (dict): per url dictionary of manifest, spec and client.
        config (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
          plugins (dict): per url dictionary of BaseOpenAIPlugin instances.

    """
    plugins = {}
    for url, manifest_spec_client in manifests_specs_clients.items():
        plugins[url] = BaseOpenAIPlugin(manifest_spec_client)
    return plugins


def scan_plugins(config: Config) -> List[AutoGPTPluginTemplate]:
    """Scan the plugins directory for plugins and loads them.

    Args:
        config (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        List[Tuple[str, Path]]: List of plugins.
    """
    loaded_plugins = []
    # Generic plugins
    plugins_path = Path(config.plugins_dir)

    plugins_config = config.plugins_config
    # Directory-based plugins
    for plugin_path in [f.path for f in os.scandir(config.plugins_dir) if f.is_dir()]:
        # Avoid going into __pycache__ or other hidden directories
        if plugin_path.startswith("__"):
            continue

        plugin_module_path = plugin_path.split(os.path.sep)
        plugin_module_name = plugin_module_path[-1]
        qualified_module_name = ".".join(plugin_module_path)

        try:
            __import__(qualified_module_name)
        except ImportError:
            logger.error(f"Failed to load {qualified_module_name}")
            continue
        plugin = sys.modules[qualified_module_name]

        if not plugins_config.is_enabled(plugin_module_name):
            logger.warning(
                f"Plugin folder {plugin_module_name} found but not configured. "
                "If this is a legitimate plugin, please add it to plugins_config.yaml "
                f"(key: {plugin_module_name})."
            )
            continue

        for _, class_obj in inspect.getmembers(plugin):
            if (
                hasattr(class_obj, "_abc_impl")
                and AutoGPTPluginTemplate in class_obj.__bases__
            ):
                loaded_plugins.append(class_obj())

    # Zip-based plugins
    for plugin in plugins_path.glob("*.zip"):
        if moduleList := inspect_zip_for_modules(str(plugin)):
            for module in moduleList:
                plugin = Path(plugin)
                module = Path(module)
                logger.debug(f"Zipped Plugin: {plugin}, Module: {module}")
                zipped_package = zipimporter(str(plugin))
                try:
                    zipped_module = zipped_package.load_module(str(module.parent))
                except ZipImportError:
                    logger.error(f"Failed to load {str(module.parent)}")
                    continue

                for key in dir(zipped_module):
                    if key.startswith("__"):
                        continue

                    a_module = getattr(zipped_module, key)
                    if not inspect.isclass(a_module):
                        continue

                    if (
                        issubclass(a_module, AutoGPTPluginTemplate)
                        and a_module.__name__ != "AutoGPTPluginTemplate"
                    ):
                        plugin_name = a_module.__name__
                        plugin_configured = plugins_config.get(plugin_name) is not None
                        plugin_enabled = plugins_config.is_enabled(plugin_name)

                        if plugin_configured and plugin_enabled:
                            logger.debug(
                                f"Loading plugin {plugin_name}. "
                                "Enabled in plugins_config.yaml."
                            )
                            loaded_plugins.append(a_module())
                        elif plugin_configured and not plugin_enabled:
                            logger.debug(
                                f"Not loading plugin {plugin_name}. "
                                "Disabled in plugins_config.yaml."
                            )
                        elif not plugin_configured:
                            logger.warning(
                                f"Not loading plugin {plugin_name}. "
                                f"No entry for '{plugin_name}' in plugins_config.yaml. "
                                "Note: Zipped plugins should use the class name "
                                f"({plugin_name}) as the key."
                            )
                    else:
                        if (
                            module_name := getattr(a_module, "__name__", str(a_module))
                        ) != "AutoGPTPluginTemplate":
                            logger.debug(
                                f"Skipping '{module_name}' because it doesn't subclass "
                                "AutoGPTPluginTemplate."
                            )

    # OpenAI plugins
    if config.plugins_openai:
        manifests_specs = fetch_openai_plugins_manifest_and_spec(config)
        if manifests_specs.keys():
            manifests_specs_clients = initialize_openai_plugins(manifests_specs, config)
            for url, openai_plugin_meta in manifests_specs_clients.items():
                if not plugins_config.is_enabled(url):
                    plugin_name = openai_plugin_meta["manifest"]["name_for_model"]
                    logger.warning(
                        f"OpenAI Plugin {plugin_name} found but not configured"
                    )
                    continue

                plugin = BaseOpenAIPlugin(openai_plugin_meta)
                loaded_plugins.append(plugin)

    if loaded_plugins:
        logger.info(f"\nPlugins found: {len(loaded_plugins)}\n" "--------------------")
    for plugin in loaded_plugins:
        logger.info(f"{plugin._name}: {plugin._version} - {plugin._description}")
    return loaded_plugins
