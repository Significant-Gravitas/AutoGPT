"""Handles loading of plugins."""
import importlib
import json
import os
import zipfile
import openapi_python_client
import requests
import abc

from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse
from zipimport import zipimporter
from openapi_python_client.cli import Config as OpenAPIConfig
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from abstract_singleton import AbstractSingleton, Singleton


from autogpt.config import Config

PromptGenerator = TypeVar("PromptGenerator")


class Message(TypedDict):
    role: str
    content: str


class BaseOpenAIPlugin():
    """
    This is a template for Auto-GPT plugins.
    """

    def __init__(self, manifests_specs_clients: dict):
        # super().__init__()
        self._name = manifests_specs_clients["manifest"]["name_for_model"]
        self._version = manifests_specs_clients["manifest"]["schema_version"]
        self._description = manifests_specs_clients["manifest"]["description_for_model"]
        self.client = manifests_specs_clients["client"]
        self.manifest = manifests_specs_clients["manifest"]
        self.openapi_spec = manifests_specs_clients["openapi_spec"]

    def can_handle_on_response(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_response method.
        Returns:
            bool: True if the plugin can handle the on_response method."""
        return False

    def on_response(self, response: str, *args, **kwargs) -> str:
        """This method is called when a response is received from the model."""
        pass

    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.
        Returns:
            bool: True if the plugin can handle the post_prompt method."""
        return False

    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.
        Args:
            prompt (PromptGenerator): The prompt generator.
        Returns:
            PromptGenerator: The prompt generator.
        """
        pass

    def can_handle_on_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_planning method.
        Returns:
            bool: True if the plugin can handle the on_planning method."""
        return False

    def on_planning(
            self, prompt: PromptGenerator, messages: List[Message]
    ) -> Optional[str]:
        """This method is called before the planning chat completion is done.
        Args:
            prompt (PromptGenerator): The prompt generator.
            messages (List[str]): The list of messages.
        """
        pass

    def can_handle_post_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_planning method.
        Returns:
            bool: True if the plugin can handle the post_planning method."""
        return False

    def post_planning(self, response: str) -> str:
        """This method is called after the planning chat completion is done.
        Args:
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        pass

    def can_handle_pre_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_instruction method.
        Returns:
            bool: True if the plugin can handle the pre_instruction method."""
        return False

    def pre_instruction(self, messages: List[Message]) -> List[Message]:
        """This method is called before the instruction chat is done.
        Args:
            messages (List[Message]): The list of context messages.
        Returns:
            List[Message]: The resulting list of messages.
        """
        pass

    def can_handle_on_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_instruction method.
        Returns:
            bool: True if the plugin can handle the on_instruction method."""
        return False

    def on_instruction(self, messages: List[Message]) -> Optional[str]:
        """This method is called when the instruction chat is done.
        Args:
            messages (List[Message]): The list of context messages.
        Returns:
            Optional[str]: The resulting message.
        """
        pass

    def can_handle_post_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_instruction method.
        Returns:
            bool: True if the plugin can handle the post_instruction method."""
        return False

    def post_instruction(self, response: str) -> str:
        """This method is called after the instruction chat is done.
        Args:
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        pass

    def can_handle_pre_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_command method.
        Returns:
            bool: True if the plugin can handle the pre_command method."""
        return False

    def pre_command(
            self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """This method is called before the command is executed.
        Args:
            command_name (str): The command name.
            arguments (Dict[str, Any]): The arguments.
        Returns:
            Tuple[str, Dict[str, Any]]: The command name and the arguments.
        """
        pass

    def can_handle_post_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_command method.
        Returns:
            bool: True if the plugin can handle the post_command method."""
        return False

    def post_command(self, command_name: str, response: str) -> str:
        """This method is called after the command is executed.
        Args:
            command_name (str): The command name.
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        pass

    def can_handle_chat_completion(
            self, messages: Dict[Any, Any], model: str, temperature: float, max_tokens: int
    ) -> bool:
        """This method is called to check that the plugin can
          handle the chat_completion method.
        Args:
            messages (List[Message]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.
          Returns:
              bool: True if the plugin can handle the chat_completion method."""
        return False

    def handle_chat_completion(
            self, messages: List[Message], model: str, temperature: float, max_tokens: int
    ) -> str:
        """This method is called when the chat completion is done.
        Args:
            messages (List[Message]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.
        Returns:
            str: The resulting response.
        """
        pass


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


def write_dict_to_json_file(data: dict, file_path: str):
    """
    Write a dictionary to a JSON file.
    Args:
        data (dict): Dictionary to write.
        file_path (str): Path to the file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def fetch_openai_plugins_manifest_and_spec(cfg: Config) -> dict:
    """
    Fetch the manifest for a list of OpenAI plugins.
        Args:
        urls (List): List of URLs to fetch.
    Returns:
        dict: per url dictionary of manifest and spec.
    """
    # TODO add directory scan
    manifests = {}
    for url in cfg.plugins_openai:
        openai_plugin_client_dir = f"{cfg.plugins_dir}/openai/{urlparse(url).netloc}"
        create_directory_if_not_exists(openai_plugin_client_dir)
        if not os.path.exists(f'{openai_plugin_client_dir}/ai-plugin.json'):
            try:
                response = requests.get(f"{url}/.well-known/ai-plugin.json")
                if response.status_code == 200:
                    manifest = response.json()
                    if manifest["schema_version"] != "v1":
                        print(f"Unsupported manifest version: {manifest['schem_version']} for {url}")
                        continue
                    if manifest["api"]["type"] != "openapi":
                        print(f"Unsupported API type: {manifest['api']['type']} for {url}")
                        continue
                    write_dict_to_json_file(manifest, f'{openai_plugin_client_dir}/ai-plugin.json')
                else:
                    print(f"Failed to fetch manifest for {url}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error while requesting manifest from {url}: {e}")
        else:
            print(f"Manifest for {url} already exists")
            manifest = json.load(open(f'{openai_plugin_client_dir}/ai-plugin.json'))
        if not os.path.exists(f'{openai_plugin_client_dir}/openapi.json'):
            openapi_spec = openapi_python_client._get_document(url=manifest["api"]["url"], path=None, timeout=5)
            write_dict_to_json_file(openapi_spec, f'{openai_plugin_client_dir}/openapi.json')
        else:
            print(f"OpenAPI spec for {url} already exists")
            openapi_spec = json.load(open(f'{openai_plugin_client_dir}/openapi.json'))
        manifests[url] = {
            'manifest': manifest,
            'openapi_spec': openapi_spec
        }
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
            print(f"Created directory: {directory_path}")
            return True
        except OSError as e:
            print(f"Error creating directory {directory_path}: {e}")
            return False
    else:
        print(f"Directory {directory_path} already exists")
        return True


def initialize_openai_plugins(manifests_specs: dict, cfg: Config, debug: bool = False) -> dict:
    """
    Initialize OpenAI plugins.
    Args:
        manifests_specs (dict): per url dictionary of manifest and spec.
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
        dict: per url dictionary of manifest, spec and client.
    """
    openai_plugins_dir = f'{cfg.plugins_dir}/openai'
    if create_directory_if_not_exists(openai_plugins_dir):
        for url, manifest_spec in manifests_specs.items():
            openai_plugin_client_dir = f'{openai_plugins_dir}/{urlparse(url).hostname}'
            _meta_option = openapi_python_client.MetaType.SETUP,
            _config = OpenAPIConfig(**{
                'project_name_override': 'client',
                'package_name_override': 'client',
            })
            prev_cwd = Path.cwd()
            os.chdir(openai_plugin_client_dir)
            Path('ai-plugin.json')
            if not os.path.exists('client'):
                client_results = openapi_python_client.create_new_client(
                    url=manifest_spec['manifest']['api']['url'],
                    path=None,
                    meta=_meta_option,
                    config=_config,
                )
                if client_results:
                    print(f"Error creating OpenAPI client: {client_results[0].header} \n"
                          f" details: {client_results[0].detail}")
                    continue
            spec = importlib.util.spec_from_file_location('client', 'client/client/client.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            client = module.Client(base_url=url)
            os.chdir(prev_cwd)
            manifest_spec['client'] = client
    return manifests_specs


def instantiate_openai_plugin_clients(manifests_specs_clients: dict, cfg: Config, debug: bool = False) -> dict:
    """
    Instantiates BaseOpenAIPluginClient instances for each OpenAI plugin.
    Args:
        manifests_specs_clients (dict): per url dictionary of manifest, spec and client.
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
          plugins (dict): per url dictionary of BaseOpenAIPluginClient instances.

    """
    plugins = {}
    for url, manifest_spec_client in manifests_specs_clients.items():
        plugins[url] = BaseOpenAIPluginClient(
            manifest=manifest_spec_client['manifest'],
            openapi_spec=manifest_spec_client['openapi_spec'],
            client=manifest_spec_client['client'],
            cfg=cfg,
            debug=debug
        )
    return plugins


def scan_plugins(cfg: Config, debug: bool = False) -> List[Tuple[str, Path]]:
    """Scan the plugins directory for plugins.

    Args:
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        List[Tuple[str, Path]]: List of plugins.
    """
    plugins = []
    # Generic plugins
    plugins_path_path = Path(cfg.plugins_dir)
    for plugin in plugins_path_path.glob("*.zip"):
        if module := inspect_zip_for_module(str(plugin), debug):
            plugins.append((module, plugin))
    # OpenAI plugins
    if cfg.plugins_openai:
        manifests_specs = fetch_openai_plugins_manifest_and_spec(cfg)
        if manifests_specs.keys():
            manifests_specs_clients = initialize_openai_plugins(manifests_specs, cfg, debug)
            for url, openai_plugin_meta in manifests_specs_clients.items():
                plugin = BaseOpenAIPlugin(openai_plugin_meta)
                plugins.append((plugin, url))


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
        cfg (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
        List[AbstractSingleton]: List of plugins initialized.
    """
    plugins = scan_plugins(cfg)
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
    return blacklist_whitelist_check(plugin_modules, cfg)
