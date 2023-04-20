# `plugins.py` documentation
This module includes functions related to loading and handling of plugins. 

## `inspect_zip_for_module(zip_path: str, debug: bool = False) -> Optional[str]`
- Inspect the given "zip_path" zipfile for a module.
- It returns the name of the module if found, otherwise, it returns `None`.

## `write_dict_to_json_file(data: dict, file_path: str) -> None`
- Writes the given dictionary data to a JSON file at the specified `file_path`.

## `fetch_openai_plugins_manifest_and_spec(cfg: Config) -> dict`
- Fetches and returns a manifest for a list of OpenAI plugins. 
- It receives a `Config` instance and tries to download manifest for each OpenAI plugin specified in the instance. 

## `create_directory_if_not_exists(directory_path: str) -> bool`
- Creates the directory at the given "directory_path" if it doesn't already exist. 
- Returns `True` if the directory is created successfully, `False` otherwise. 

## `initialize_openai_plugins(manifests_specs: dict, cfg: Config, debug: bool = False) -> dict`
- Initializes an OpenAI plugin. 
- It receives a "manifests_specs" dictionary, a `Config` instance and returns the initialized plugin.

## `instantiate_openai_plugin_clients(manifests_specs_clients: dict, cfg: Config, debug: bool = False) -> dict`
- Instantiates `BaseOpenAIPlugin` instances for each OpenAI plugin.
- `manifests_specs_clients`: per URL dictionary of manifest, specification, and client.
- `cfg`: Configuration instance containing plugin configurations.

## `scan_plugins(cfg: Config, debug: bool = False) -> List[AutoGPTPluginTemplate]`
- Scans the plugins directory for plugins and loads them. 
- The function scans the following:
    1. Generic plugins
    2. OpenAI plugins
- The function receives a `Config` instance and returns a list of loaded plugins.

## `denylist_allowlist_check(plugin_name: str, cfg: Config) -> bool`
- Checks if the plugin is in the allowlist or denylist. 
- Receives the name of the plugin and a `Config` instance.
- Returns `True` if the plugin is allowed, `False` otherwise.