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
import yaml
from openapi_python_client import GeneratorError
from openapi_python_client.cli import Config as OpenAPIConfig
from pydantic import BaseModel

from autogpt.commands.command import CommandRegistry, Command
from autogpt.config import Config

OPENAPI_COMMANDS_REGISTRY = str(Path(os.getcwd()) / "openapi_commands.yaml")


class AutoGPTOpenAPIConfig(BaseModel):
    url: str
    openapi_spec: Optional[str] = None
    openai_manifest: Optional[str] = None
    enabled: bool
    auth: dict


def load_openapi_config(cfg: Config, openapi_registry_path: str = OPENAPI_COMMANDS_REGISTRY) -> dict[
    AutoGPTOpenAPIConfig]:
    """
    Load the openapi commands registry and create the command registry.
    Args:
        cfg (Config): The configuration object.
        openapi_registry_path: path to registry yaml file
    Returns:
        dict: enabled_openapi_apis - A dictionary of enabled openapi apis configs.
    """

    try:
        with open(openapi_registry_path, encoding="utf-8") as file:
            openapi_registry = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        return {}

    enabled_openapi_apis = {}
    for api_name, api_config in openapi_registry.items():
        if api_name in cfg.openapi_apis and api_config["enabled"]:
            enabled_openapi_apis[api_name] = AutoGPTOpenAPIConfig(**api_config)

    return enabled_openapi_apis


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


def get_openapi_spec(openapi_spec_url: str, openapi_plugin_client_dir: str) -> Optional[dict]:
    """
    Fetch the OpenAPI(Swagger spec for a given url if it does not exist in file yet.
    Args:
        openapi_spec_url (str): URL to the OpenAPI spec.
        openapi_plugin_client_dir (str): Path to the directory where the spec is stored.
    Returns:
        dict: OpenAPI spec.
    """
    if not os.path.exists(f"{openapi_plugin_client_dir}/openapi.json"):
        openapi_spec = openapi_python_client._get_document(
            url=openapi_spec_url, path=None, timeout=5
        )
        if isinstance(openapi_spec, GeneratorError):
            print(f"Error fetching OpenAPI spec for {openapi_spec_url}: {openapi_spec.header}")
            return
        write_dict_to_json_file(
            openapi_spec, f"{openapi_plugin_client_dir}/openapi.json"
        )
    else:
        print(f"OpenAPI spec for {openapi_spec_url} already exists")
        openapi_spec = json.load(open(f"{openapi_plugin_client_dir}/openapi.json"))
    return openapi_spec


def generate_openai_manifest_from_openapi_spec(api_name: str, api_config: AutoGPTOpenAPIConfig,
                                               openapi_spec: dict) -> dict:
    """
    Generate an OpenAI manifest from an OpenAPI spec.
    Args:
        openapi_spec: OpenAPI spec.
    Returns:
        dict: OpenAI manifest.
    """
    return {
        "schema_version": "v1",
        "name_for_model":
            openapi_spec['info']['title'].replace(" ", "") if openapi_spec['info'].get('title') else api_name,
        "name_for_human":
            openapi_spec['info']['title'].replace(" ", "") if openapi_spec['info'].get('title') else api_name,
        "description_for_human":
            openapi_spec['info']['description'] if openapi_spec['info'].get('description') else api_config.url,
        "description_for_model":
            openapi_spec['info']['description'] if openapi_spec['info'].get('description') else api_config.url,
        "api": {
            "type": "openapi",
            "url": api_config.url + api_config.openapi_spec,
            "has_user_authentication": False
        },
        "auth": {
            "type": "none"
        },
        "logo_url": "",
        "contact_email": "",
        "legal_info_url": ""
    }


def get_openai_manifest(openapi_manifest_url: str,
                        openapi_plugin_client_dir: str) -> Optional[dict]:
    """
    Fetch the OpenAI manifest for a given url if it does not exist in file yet.
    For OpenAI Plugins Compatibility, we need to fetch/generate a manifest from the spec.
    Args:
        openapi_manifest_url: URL to the OpenAI manifest.
        openapi_spec: OpenAPI spec.
        openapi_plugin_client_dir: Path to the directory where the manifest is stored.

    Returns:
        dict: OpenAI manifest.
    """
    if not os.path.exists(f"{openapi_plugin_client_dir}/ai-plugin.json"):
        manifest = None
        try:
            response = requests.get(openapi_manifest_url)
            if response.status_code == 200:
                manifest = response.json()
                write_dict_to_json_file(
                    manifest, f"{openapi_plugin_client_dir}/ai-plugin.json"
                )
            else:
                print(f"Failed to fetch manifest for {openapi_manifest_url}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error while requesting manifest from {openapi_manifest_url}: {e}")
    else:
        print(f"Manifest for {openapi_manifest_url} already exists")
        manifest = json.load(open(f"{openapi_plugin_client_dir}/ai-plugin.json"))
    return manifest


def get_openapi_specs_and_manifests(openapi_config: dict[AutoGPTOpenAPIConfig], plugins_dir: str) -> dict:
    """
    Fetch the manifest for a list of OpenAPI plugins.
    Args:
        cfg (Config): Config instance including plugins config
    Returns:
        dict: per url dictionary of manifest and spec.
    """
    specs_manifests = {}

    for api_name, api_config in openapi_config.items():
        openapi_plugin_client_dir = f"{plugins_dir}/openapi/{api_name}"
        create_directory_if_not_exists(openapi_plugin_client_dir)

        openapi_spec_url = api_config.url + api_config.openapi_spec if api_config.openapi_spec else None
        openai_manifest_url = api_config.url + api_config.openai_manifest if api_config.openai_manifest else None

        openapi_spec = get_openapi_spec(openapi_spec_url, openapi_plugin_client_dir) if openapi_spec_url else None
        manifest = get_openai_manifest(openai_manifest_url, openapi_plugin_client_dir) if openai_manifest_url else None

        if not openapi_spec and manifest:
            openapi_spec = get_openapi_spec(manifest["api"]["url"], openapi_plugin_client_dir)

        if openapi_spec and not manifest:
            manifest = generate_openai_manifest_from_openapi_spec(api_name, api_config, openapi_spec)

        specs_manifests[api_name] = {"manifest": manifest, "openapi_spec": openapi_spec}

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
        specs_manifests: dict, openapi_config: dict[AutoGPTOpenAPIConfig], plugins_dir: str
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
    openapi_plugins_dir = f"{plugins_dir}/openapi"
    if create_directory_if_not_exists(openapi_plugins_dir):
        for api_name, manifest_spec in specs_manifests.items():
            openai_plugin_client_dir = f"{openapi_plugins_dir}/{api_name}"
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
            client = client_dynamic_package.Client(base_url=openapi_config[api_name].url, follow_redirects=True)
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


def instantiate_openapi_clients(openapi_config: dict[AutoGPTOpenAPIConfig], cfg: Config) -> Optional[dict]:
    """Scan the plugins directory for plugins and loads them.

    Args:
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        List[Tuple[str, Path]]: List of plugins.
    """
    manifests_specs = get_openapi_specs_and_manifests(openapi_config, cfg.plugins_dir)
    if manifests_specs.keys():
        return initialize_openapi_plugins(manifests_specs, openapi_config, cfg.plugins_dir)
    return


def import_openapi_apis_as_commands(command_registry: CommandRegistry, cfg: Config) -> CommandRegistry:
    """
    Import OpenAPI APIs as commands.
    Args:
        command_registry (CommandRegistry): Command registry instance.
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
        CommandRegistry: Command registry instance.
    """
    openapi_config = load_openapi_config(cfg)
    if not openapi_config:
        return command_registry
    openapi_commands_specs = instantiate_openapi_clients(openapi_config, cfg)
    for openapi_commands_spec in openapi_commands_specs.values():
        for method_name, method in openapi_commands_spec['modules'].items():
            command = Command(
                name=method_name,
                description=f"{method['description']}. From OpenAPI: {openapi_commands_spec['manifest']['description_for_model']}",
                method=method['method'],
                signature=method['signature']
            )

            command_registry.register(command)
    return command_registry
