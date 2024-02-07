from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import pkgutil
import time
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterator

from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.configs.schema import Configurable, SystemConfiguration

# from AFAAS.core.tools.builtins import BUILTIN_TOOLS
from AFAAS.core.tools.tool_decorator import TOOL_WRAPPER_MARKER
from AFAAS.interfaces.adapters import (
    AbstractChatModelProvider,
    CompletionModelFunction,
    ModelProviderName,
)
from AFAAS.interfaces.db.db import AbstractMemory
from AFAAS.interfaces.tools.base import (
    AFAASBaseTool,
    AbstractToolRegistry,
    ToolConfiguration,
)
from AFAAS.interfaces.tools.schema import ToolResult
from AFAAS.interfaces.workspace import AbstractFileWorkspace
from AFAAS.lib.sdk.cache_manager import CacheManager


class ToolsRegistryConfiguration(SystemConfiguration):
    """
    Configuration for the ToolRegistry subsystem.

    Attributes:
        tools: A dictionary mapping tool names to their configurations.
    """

    tools: dict[str, ToolConfiguration] = {}


ToolsRegistryConfiguration.model_rebuild()


class DefaultToolRegistry(Configurable, AbstractToolRegistry):
    """
    A manager for a collection of Tool objects. Supports registration, modification, retrieval, and loading
    of tool plugins from a specified directory.

    Attributes:
        default_settings: Default system settings for the DefaultToolRegistry.
    """

    class SystemSettings(Configurable.SystemSettings):
        """
        System settings for ToolRegistry.

        Attributes:
            configuration: Configuration settings for ToolRegistry.
        """

        configuration: ToolsRegistryConfiguration = ToolsRegistryConfiguration()
        name: str = "simple_tool_registry"
        description: str = "A simple tool registry."

    def __init__(
        self,
        settings: DefaultToolRegistry.SystemSettings,
        db: AbstractMemory,
        workspace: AbstractFileWorkspace,
        model_providers: dict[ModelProviderName, AbstractChatModelProvider],
        include_builtins=True,
    ):
        """
        Initialize the DefaultToolRegistry.

        Args:
            settings: Configuration settings for the registry.
            logger: Logging instance to use for the registry.
            db: Memory instance for the registry.
            workspace: Workspace instance for the registry.
            model_providers: A dictionary mapping model provider names to chat model providers.

        Example:
            registry = DefaultToolRegistry(settings, logger, db, workspace, model_providers)
        """

        LOG.debug("Initializing DefaultToolRegistry...")
        LOG.notice(
            "Memory, Workspace and ModelProviders are not used anymore argument are supported"
        )

        # self._db = db
        # self._workspace = workspace
        # self._model_providers = model_providers
        # self.tools: list[Tool] = []
        # for (
        #     tool_name,
        #     tool_configuration,
        # ) in self._settings.configuration.tools.items():
        #     self.register_tool(tool_name, tool_configuration)

        self.tools_by_name: dict[AFAASBaseTool] = {}
        # self.tool_aliases : dict[AbstractTool]  = {}
        self._tool_module: dict[AFAASBaseTool] = {}

        self.initialize_cache()

        if include_builtins:
            from AFAAS.core.tools.builtins import BUILTIN_MODULES

            for module in BUILTIN_MODULES:
                self._import_tool_module(module_name=module)

    last_updated = 0
    plugin_directory = "AFAAS/plugins/tools/"
    cache_manager = CacheManager()
    categories = {}

    def initialize_cache(self):
        current_time = time.time()
        last_updated_in_cache = self.cache_manager.get_cache_time() or 0
        # Determine if the cache needs to be updated
        needs_update = any(
            os.path.getmtime(os.path.join(self.plugin_directory, f))
            > last_updated_in_cache
            for f in os.listdir(self.plugin_directory)
            if f.endswith(".py") and not f.startswith("__")
        )

        if needs_update:
            self.rebuild_cache()
            self.last_updated = current_time

    def rebuild_cache(self):
        self.cache_manager.clear_cache()
        for filename in os.listdir(self.plugin_directory):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_path = (
                    f"{self.plugin_directory.replace('/', '.')}{filename[:-3]}"
                )
                module = self._import_tool_module(module_path)
                for name, attr in inspect.getmembers(module):
                    if hasattr(attr, TOOL_WRAPPER_MARKER):  # Check if attr is a tool
                        tool_instance = getattr(attr, "tool")
                        for category in tool_instance.categories:
                            existing_modules = self.cache_manager.get(category) or {}
                            existing_modules.update({tool_instance.name: module_path})
                            self.cache_manager.set(category, existing_modules)

        self.cache_manager.set_cache_time()

    def add_all_tool_categories(self):
        # Retrieve all categories from the cache
        all_categories = self.cache_manager.get_all_categories()

        # Add each category to the registry
        for category in all_categories:
            self.add_tool_category(category)

    def add_tool_categories(self, categories: list[str]):
        for category in categories:
            self.add_tool_category(category)

    def add_tool_category(self, category: str):
        tool_modules = self.cache_manager.get(category) or {}
        for tool_name, module_path in tool_modules.items():
            self._add_tool_module(tool_name, module_path)

    def _add_tool_module(self, tool_name: str, module_path: str) -> None:
        if tool_name not in self._tool_module:
            self._tool_module[tool_name] = module_path

    def get_tool(self, tool_name: str) -> AFAASBaseTool | None:
        if tool_name not in self.tools_by_name and tool_name in self._tool_module:
            self._load_tool(tool_name)
        return self.tools_by_name.get(tool_name)

    def _load_tool(self, tool_name: str):
        if tool_name in self._tool_module:
            module_path = self._tool_module[tool_name]
            module = self._import_tool_module(module_path)
        else:
            LOG.error(f"Tool '{tool_name}' not found in cache.")

    def __contains__(self, tool_name: str):
        return tool_name in self.tools_by_name  # or tool_name in self.tool_aliases

    def register_tool(
        self,
        tool_name: str,
        tool_configuration: ToolConfiguration,
        aliases: list[str] = [],
        category: str = None,
    ) -> None:
        return self.register(AFAASBaseTool(tool_configuration))

    #     """
    #     Register a new tool with the registry.

    #     Args:
    #     - tool_name (str): Name of the tool.
    #     - tool_configuration (ToolConfiguration): Configuration details for the tool.
    #     - aliases (list[str], optional): A list of alternative names for the tool. Defaults to an empty list.
    #     - category (str, optional): Category to which the tool belongs. Defaults to None.

    #     Example:
    #     ```python
    #     registry = DefaultToolRegistry(...)
    #     registry.register_tool("sample_tool", ToolConfiguration(), aliases=["example"], category="sample_category")
    #     ```
    #     """
    #     tool_class = DefaultPluginService.get_plugin(tool_configuration.location)
    #     tool_args = {
    #         "logger": LOG.getChild(tool_name),
    #         "configuration": tool_configuration,
    #     }
    #     if tool_configuration.packages_required:
    #         # TODO: Check packages are installed and maybe install them.
    #         pass
    #     if tool_configuration.db_provider_required:
    #         tool_args["db"] = self._db
    #     if tool_configuration.workspace_required:
    #         tool_args["workspace"] = self._workspace
    #     if tool_configuration.language_model_required:
    #         tool_args["language_model_provider"] = self._model_providers[
    #             tool_configuration.language_model_required.provider_name
    #         ]
    #     tool = tool_class(**tool_args)
    #     self.tools.append(tool)

    #     for alias in aliases:
    #         if alias in self.tool_aliases:
    #             # Handle overwriting aliases or log a warning
    #             logging.warning(f"Alias '{alias}' is already in use.")
    #         else:
    #             self.tool_aliases[alias] = tool_name

    #     # Handle categorization
    #     if category:
    #         if category not in self.categories:
    #             self.categories[category] = self.ToolCategory(
    #                 name=category, title=category.capitalize(), description=""
    #             )
    #         self.categories[category].tools.append(tool_name)

    def register(self, tool: AFAASBaseTool) -> None:
        if tool.name in self.tools_by_name:
            LOG.notice(
                f"Tool '{tool.name}' already registered and will be overwritten!"
            )
        self.tools_by_name[tool.name] = tool

        # if tool.name in self.tool_aliases:
        #     LOG.warning(
        #         f"Tool '{tool.name}' will overwrite alias with the same name of "
        #         f"'{self.tool_aliases[tool.name]}'!"
        #     )
        # for alias in tool.aliases:
        #     self.tool_aliases[alias] = tool

    def unregister(self, tool: AFAASBaseTool) -> None:
        if tool.name in self.tools_by_name:
            del self.tools_by_name[tool.name]
            # for alias in tool.aliases:
            #     del self.tool_aliases[alias]
        else:
            raise KeyError(f"Tool '{tool.name}' not found in registry.")

    # def reload_tools(self) -> None:
    #     for cmd_name in self.tools:
    #         cmd = self.tools[cmd_name]
    #         module = self._import_module(cmd.__module__)
    #         reloaded_module = self._reload_module(module)
    #         if hasattr(reloaded_module, "register"):
    #             reloaded_module.register(self)

    def dump_tools(self, available=None) -> list[CompletionModelFunction]:
        if available is not None:
            LOG.warning("Parameter `available` not implemented")

        function_list: list[CompletionModelFunction] = []

        for tool in self.get_tool_list():
            function_list.append(tool.dump())

        return function_list

    async def perform(self, tool_name: str, **kwargs) -> ToolResult:
        LOG.warning("### FUNCTION DEPRECATED !!! ###")
        tool = self.get_tool(tool_name)
        if tool:
            if tool.is_async:
                return await tool(**kwargs)
            else:
                return tool(**kwargs)
        raise KeyError(f"Tool '{tool_name}' not found in registry")

    def call(self, tool_name: str, agent: BaseAgent, **kwargs) -> Any:
        if tool := self.get_tool(tool_name):
            return tool(**kwargs, agent=agent)
        raise KeyError(f"Tool '{tool_name}' not found in registry")

    def list_tools_descriptions(self) -> list[str]:
        LOG.warning("Function deprecated")
        return [f"{tool.name}: {tool.description}" for tool in self.get_tool_list()]

    def get_tools_names(self) -> list[str]:
        available_tools = []
        for tool_name in self._tool_module:
            tool = self.get_tool(tool_name)
            if tool is not None:
                available_tools.append(tool.name)
        return available_tools

    def get_tool_list(self) -> list[AFAASBaseTool]:
        available_tools = []
        for tool_name in self._tool_module:
            tool = self.get_tool(tool_name)
            if tool is not None:
                available_tools.append(tool)
        return available_tools

    def _import_tool_module(self, module_name: str) -> None:
        module = importlib.import_module(module_name)

        # category = self.register_module_category(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            tool = None

            # Register decorated functions
            if getattr(attr, TOOL_WRAPPER_MARKER, False):
                tool = attr.tool

            # Register tool classes
            elif (
                inspect.isclass(attr)
                and attr.__module__ != 'AFAAS.core.tools.tool'
                and issubclass(attr, AFAASBaseTool)
                and attr != AFAASBaseTool
            ):
                tool = attr()

            if tool:
                self.register(tool)
                # NOTE: Keeping it but not sure we need to populate a list of category
                # for category in tool.categories:
                #     if category not in self.categories:
                #         # Initialize the category if it doesn't exist
                #         self.categories[category] = []
                #     # Append the tool name to the category
                #     self.categories[category].append(tool.name)

        return module

    async def call(self, tool_name: str, **kwargs) -> ToolResult:
        LOG.warning("ToolRegistry.call() is deprecated")

        return await self.perform(tool_name=tool_name, **kwargs)

    @staticmethod
    def write_and_load_module_in_afaas(module_name, code):
        # Find the base directory of the AFAAS package
        afaas_package = pkgutil.get_loader("AFAAS")
        if not afaas_package:
            raise ImportError("AFAAS package not found")

        afaas_path = afaas_package.get_filename()

        # Assuming the structure is AFAAS/__init__.py
        base_directory = os.path.dirname(afaas_path)

        # Target directory for the new module
        target_directory = os.path.join(base_directory, "plugins", "tools")

        # Ensure the target directory exists
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Path for the new Python file
        file_path = os.path.join(target_directory, module_name + ".py")

        # Write the Python code to the file
        with open(file_path, "w") as file:
            file.write(code)

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module, file_path
