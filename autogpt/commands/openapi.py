"""Handles any openapi API spec"""

import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Any
from urllib.parse import urlparse

import openapi_python_client
import requests
from openapi_python_client.cli import Config as OpenAPIConfig

from autogpt.commands.command import CommandRegistry, Command
from autogpt.config import Config


def write_dict_to_json_file(data: dict, file_path: str) -> None:
    """
    Write a dictionary to a JSON file. For writing manifests and specs.
    Args:
        data (dict): Dictionary to write.
        file_path (str): Path to the file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def create_directory_if_not_exists(directory_path: str) -> bool:
    """
    Create a directory if it does not exist for storing OpenAPI plugins.
    Args:
        directory_path (str): Path to the directory.
    Returns:
        bool: True if the directory was created, else False.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
            return True
        except OSError as e:
            print(f"Error creating directory {directory_path}: {e}")
            return False
    else:
        print(f"Directory {directory_path} already exists")
        return True


def fetch_openapi_manifests_and_specs(cfg: Config) -> dict:
    """
    Fetch the manifest for a list of OpenAPI plugins.
        Args:
        cfg (Config): Config instance including plugins config
    Returns:
        dict: per url dictionary of manifest and spec.
    """
    specs_manifests = {}
    for url in cfg.openapi_apis:
        openapi_plugin_client_dir = f"{cfg.plugins_dir}/openai/{urlparse(url).netloc}"
        create_directory_if_not_exists(openapi_plugin_client_dir)
        # For OpenAI Plugins Compatibility, we need to fetch/generate a manifest from the spec
        # Check if we already have OpenAI manifest fetched/stored or generated, load from file if so
        if not os.path.exists(f"{openapi_plugin_client_dir}/ai-plugin.json"):
            try:
                response = requests.get(f"{url}/.well-known/ai-plugin.json")
                if response.status_code == 200:
                    manifest = response.json()
                    if manifest["schema_version"] != "v1":
                        print(
                            f"Unsupported manifest version: {manifest['schem_version']} for {url}"
                        )
                        continue
                    if manifest["api"]["type"] != "openapi":
                        print(
                            f"Unsupported API type: {manifest['api']['type']} for {url}"
                        )
                        continue
                    write_dict_to_json_file(
                        manifest, f"{openapi_plugin_client_dir}/ai-plugin.json"
                    )
                else:
                    print(f"Failed to fetch manifest for {url}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error while requesting manifest from {url}: {e}")
            # TODO Add Case for generating manifest from OpenAPI spec
            manifest = {}
        else:
            print(f"Manifest for {url} already exists")
            manifest = json.load(open(f"{openapi_plugin_client_dir}/ai-plugin.json"))
        # Check if we already have OpenAPI(Swagger) spec fetched and stored, load from file if so
        if not os.path.exists(f"{openapi_plugin_client_dir}/openapi.json"):
            openapi_spec = openapi_python_client._get_document(
                url=manifest["api"]["url"], path=None, timeout=5
            )
            write_dict_to_json_file(
                openapi_spec, f"{openapi_plugin_client_dir}/openapi.json"
            )
        else:
            print(f"OpenAPI spec for {url} already exists")
            openapi_spec = json.load(open(f"{openapi_plugin_client_dir}/openapi.json"))

        specs_manifests[url] = {"manifest": manifest, "openapi_spec": openapi_spec}
    return specs_manifests


def camel_to_snake(name: str) -> str:
    """
    Convert a camel case string to snake case. For converting paths/operationId models names from OpenAPI spec.
    Args:
        name (str): Camel case string.
    Returns:
        object: str
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def initialize_openapi_plugins(
        specs_manifests: dict, cfg: Config
) -> dict:
    """
    Initialize OpenAI plugins.
    Args:
        specs_manifests (dict): per url dictionary of manifest and spec.
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
        dict: per url dictionary of manifest, spec and client.
    """
    openapi_plugins_dir = f"{cfg.plugins_dir}/openai"
    if create_directory_if_not_exists(openapi_plugins_dir):
        for url, manifest_spec in specs_manifests.items():
            openai_plugin_client_dir = f"{openapi_plugins_dir}/{urlparse(url).hostname}"
            client_name = 'client'
            _meta_option = (openapi_python_client.MetaType.SETUP,)
            _config = OpenAPIConfig(
                **{
                    "project_name_override": client_name,
                    "package_name_override": client_name,
                }
            )
            prev_cwd = Path.cwd()
            os.chdir(openai_plugin_client_dir)
            # If we have no client created, we age generating one using
            # https://pypi.org/project/openapi-python-client/
            if not os.path.exists(client_name):
                client_results = openapi_python_client.create_new_client(
                    url=manifest_spec["manifest"]["api"]["url"],
                    path=None,
                    meta=_meta_option,
                    config=_config,
                )
                if client_results:
                    print(
                        f"Error creating OpenAPI client: {client_results[0].header} \n"
                        f" details: {client_results[0].detail}"
                    )
                    continue
            # Importing generated client as a package
            sys.path.append(client_name)
            client_dynamic_package = importlib.import_module(client_name)
            client = client_dynamic_package.Client(base_url=url, follow_redirects=True)
            # for each API endpoint and method we have module generated by openapi-python-client
            # modules has names based on operationId from OpenAPI spec
            # here we are importing those modules and creating a wrapper function for each
            # method that injects client and returns JSON output which is observable by AutoGPT
            for request_method in manifest_spec["openapi_spec"]["paths"].values():
                for method, method_info in request_method.items():
                    module_name = camel_to_snake(method_info['operationId'])
                    submodule_path = f"client.api.default.{module_name}"
                    submodule = importlib.import_module(submodule_path)

                    method_signature = str(inspect.signature(submodule.sync))
                    fixed_method_signature = \
                        method_signature.replace(f"*, client: {client_name}.{client_name}.Client, ", '')

                    def openapi_method_wrapper(
                            *args: Any,
                            **kwargs: Any
                    ) -> Optional[str]:
                        """Wrapper for openapi call function that takes in a sync function,
                        client, and arbitrary arguments, and returns JSON output.

                        Args:
                            func (Callable[..., Optional[CheckWeatherUsingGETResponse200]]): The sync function.
                            client (Client): The client instance.
                            *args (Any): Positional arguments for the sync function.
                            **kwargs (Any): Keyword arguments for the sync function.

                        Returns:
                            Optional[str]: The JSON-formatted response.
                        """
                        response = submodule.sync(client=client, *args, **kwargs)
                        return json.dumps(response.to_dict()) if response is not None else None

                    manifest_spec["modules"] = {
                        module_name: {
                            'description': method_info['summary'],
                            'method': openapi_method_wrapper,
                            'signature': fixed_method_signature
                        }
                    }

            os.chdir(prev_cwd)
            manifest_spec["client"] = client
    return specs_manifests


def instantiate_openapi_clients(cfg: Config, debug: bool = False) -> Optional[dict]:
    """Scan the plugins directory for plugins and loads them.

    Args:
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        List[Tuple[str, Path]]: List of plugins.
    """
    if cfg.openapi_apis:
        manifests_specs = fetch_openapi_manifests_and_specs(cfg)
        if manifests_specs.keys():
            return initialize_openapi_plugins(manifests_specs, cfg, debug)
    return


def import_openapi_apis_as_commands(command_registry: CommandRegistry, cfg: Config, debug: bool = False) -> None:
    """
    Import OpenAPI APIs as commands.
    Args:
        command_registry (CommandRegistry): Command registry instance.
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
        CommandRegistry: Command registry instance.
    """
    openapi_commands_specs = instantiate_openapi_clients(cfg, debug)
    for openapi_commands_spec in openapi_commands_specs:
        for method_name, method in openapi_commands_spec.items():
            command = Command(
                name=method_name,
                description=f"{method['description']}. From API: {openapi_commands_spec.description}",
                method=method['method'],
                signature=method['signature']
            )

            command_registry.register(command)
    return command_registry
