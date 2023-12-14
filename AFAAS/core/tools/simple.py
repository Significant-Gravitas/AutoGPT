from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterator

from AFAAS.core.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

if TYPE_CHECKING:
    from AFAAS.core.agents.base import BaseAgent

from AFAAS.core.configuration import (Configurable,
                                                         SystemConfiguration)
from AFAAS.core.memory.base import AbstractMemory
from AFAAS.core.resource.model_providers import (
    BaseChatModelProvider, CompletionModelFunction, ModelProviderName)
from AFAAS.core.tools.base import (BaseToolsRegistry, BaseTool,
                                                      ToolConfiguration)
from AFAAS.core.tools.command_decorator import \
    AUTO_GPT_TOOL_IDENTIFIER
# from AFAAS.core.tools.builtins import BUILTIN_TOOLS
from AFAAS.core.tools.schema import ToolResult
from AFAAS.core.workspace.base import AbstractFileWorkspace


class ToolsRegistryConfiguration(SystemConfiguration):
    """
    Configuration for the ToolRegistry subsystem.

    Attributes:
        tools: A dictionary mapping tool names to their configurations.
    """

    tools: dict[str, ToolConfiguration] = {}


ToolsRegistryConfiguration.update_forward_refs()


class SimpleToolRegistry(Configurable, BaseToolsRegistry):
    """
    A manager for a collection of Tool objects. Supports registration, modification, retrieval, and loading
    of tool plugins from a specified directory.

    Attributes:
        default_settings: Default system settings for the SimpleToolRegistry.
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

    @dataclass
    class ToolCategory:
        """
        Represents a category of tools.

        Attributes:
            name: Name of the category.
            title: Display title for the category.
            description: Description of the category.
            tools: List of Tool objects associated with the category.
            modules: List of ModuleType objects related to the category.
        """

        name: str
        title: str
        description: str
        tools: list[BaseTool] = field(default_factory=list[BaseTool])
        modules: list[ModuleType] = field(default_factory=list[ModuleType])

        class Config:
            arbitrary_types_allowed = True

    def __init__(
        self,
        settings: SimpleToolRegistry.SystemSettings,
        
        memory: AbstractMemory,
        workspace: AbstractFileWorkspace,
        model_providers: dict[ModelProviderName, BaseChatModelProvider],
    ):
        """
        Initialize the SimpleToolRegistry.

        Args:
            settings: Configuration settings for the registry.
            logger: Logging instance to use for the registry.
            memory: Memory instance for the registry.
            workspace: Workspace instance for the registry.
            model_providers: A dictionary mapping model provider names to chat model providers.

        Example:
            registry = SimpleToolRegistry(settings, logger, memory, workspace, model_providers)
        """

        # self._memory = memory
        # self._workspace = workspace
        # self._model_providers = model_providers
        # self.tools: list[Tool] = []
        # for (
        #     tool_name,
        #     tool_configuration,
        # ) in self._configuration.tools.items():
        #     self.register_tool(tool_name, tool_configuration)

        self.tools = {}
        self.tool_aliases = {}
        self.categories = {}

    def __contains__(self, tool_name: str):
        return tool_name in self.tools or tool_name in self.tool_aliases

    def _import_module(self, module_name: str):
        """Imports a module using its name."""
        return importlib.import_module(module_name)

    def _reload_module(self, module):
        """Reloads a given module."""
        return importlib.reload(module)

    def register_tool(
        self,
        tool_name: str,
        tool_configuration: ToolConfiguration,
        aliases: list[str] = [],
        category: str = None,
    ) -> None:
        return self.register(BaseTool(tool_configuration))

    #     """
    #     Register a new tool with the registry.

    #     Args:
    #     - tool_name (str): Name of the tool.
    #     - tool_configuration (ToolConfiguration): Configuration details for the tool.
    #     - aliases (list[str], optional): A list of alternative names for the tool. Defaults to an empty list.
    #     - category (str, optional): Category to which the tool belongs. Defaults to None.

    #     Example:
    #     ```python
    #     registry = SimpleToolRegistry(...)
    #     registry.register_tool("sample_tool", ToolConfiguration(), aliases=["example"], category="sample_category")
    #     ```
    #     """
    #     tool_class = SimplePluginService.get_plugin(tool_configuration.location)
    #     tool_args = {
    #         "logger": LOG.getChild(tool_name),
    #         "configuration": tool_configuration,
    #     }
    #     if tool_configuration.packages_required:
    #         # TODO: Check packages are installed and maybe install them.
    #         pass
    #     if tool_configuration.memory_provider_required:
    #         tool_args["memory"] = self._memory
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

    def register(self, cmd: BaseTool) -> None:
        """
        Register a new tool with the registry.

        Args:
        - tool_name (str): Name of the tool.
        - tool_configuration (ToolConfiguration): Configuration details for the tool.
        - aliases (list[str], optional): A list of alternative names for the tool. Defaults to an empty list.
        - category (str, optional): Category to which the tool belongs. Defaults to None.

        Example:
        ```python
        registry = SimpleToolRegistry(...)
        registry.register_tool("sample_tool", ToolConfiguration(), aliases=["example"], category="sample_category")
        ```
        """
        if cmd.name in self.tools:
            LOG.warn(
                f"Tool '{cmd.name}' already registered and will be overwritten!"
            )
        self.tools[cmd.name] = cmd

        if cmd.name in self.tool_aliases:
            LOG.warn(
                f"Tool '{cmd.name}' will overwrite alias with the same name of "
                f"'{self.tool_aliases[cmd.name]}'!"
            )
        for alias in cmd.aliases:
            self.tool_aliases[alias] = cmd

    def unregister(self, tool: BaseTool) -> None:
        """
        Unregister an tool from the registry.

        Args:
        - tool (Tool): The tool instance to unregister.

        Raises:
        - KeyError: If the tool is not found in the registry.

        Example:
        ```python
        registry = SimpleToolRegistry(...)
        tool_instance = ...
        registry.unregister(tool_instance)
        ```
        """
        if tool.name in self.tools:
            del self.tools[tool.name]
            for alias in tool.aliases:
                del self.tool_aliases[alias]
        else:
            raise KeyError(f"Tool '{tool.name}' not found in registry.")

    def dump_tools(self, available=None) -> list[CompletionModelFunction]:
        if available is not None:
            LOG.warning("Parameter `available` not implemented")

        param_dict = {}
        function_list: list[CompletionModelFunction] = []

        for tool in self.tools.values():
            param_dict = {}  # Reset param_dict for each tool
            for parameter in tool.parameters:
                param_dict[parameter.name] = parameter.spec

            name = tool.name
            description = tool.description

            function_list.append(
                CompletionModelFunction(
                    name=name,
                    description=description,
                    parameters=param_dict,
                )
            )

        return function_list

    # CompletionModelFunction.parse
    def reload_tools(self) -> None:
        """
        Reloads all loaded tool plugins.

        Useful for dynamically updating tools without restarting the application.

        Example:
        ```python
        registry = SimpleToolRegistry(...)
        registry.reload_tools()
        ```
        """
        for cmd_name in self.tools:
            cmd = self.commands[cmd_name]
            module = self._import_module(cmd.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, "register"):
                reloaded_module.register(self)

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """
        Retrieve a specific tool by its name.

        Args:
        - tool_name (str): Name of the tool to retrieve.

        Returns:
        - Tool: The matched tool.

        Raises:
        - ValueError: If the tool is not found.

        Example:
        ```python
        registry = SimpleToolRegistry(...)
        tool = registry.get_tool("sample_tool")
        ```
        """
        if tool_name in self.tools:
            return self.tools[tool_name]

        if tool_name in self.tool_aliases:
            return self.tool_aliases[tool_name]

    async def perform(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Retrieve a registered tool by its name.

        Args:
            tool_name: Name or alias of the tool to retrieve.

        Returns:
            The requested Tool instance.

        Raises:
            ValueError: If no tool with the given name or alias is found.

        Example:
            tool = registry.get_tool("example_tool")
        """
        LOG.warning("### FUNCTION DEPRECATED !!! ###")
        tool = self.get_tool(tool_name)
        if tool:
            if tool.is_async == True:
                return await tool(**kwargs)
            else:
                return tool(**kwargs)

        raise KeyError(f"Tool '{tool_name}' not found in registry")

    def call(self, command_name: str, agent: BaseAgent, **kwargs) -> Any:
        if command := self.get_command(command_name):
            return command(**kwargs, agent=agent)
        raise KeyError(f"Tool '{command_name}' not found in registry")

    def list_available_tools(self, agent: BaseAgent) -> Iterator[BaseTool]:
        """Iterates over all registered tools and yields those that are available.

        Params:
            agent (BaseAgent): The agent that the commands will be checked against.


        Yields:
        - Tool: The next available tools.

        Example:
        ```python
        registry = SimpleToolRegistry(...)
        for tool in registry.list_available_tools():
            print(tool.name())
        ```
        """

        for tool in self.tools.values():
            available = tool.available
            if callable(tool.available):
                available = tool.available(agent)
            if available:
                yield tool

    def list_tools_descriptions(self) -> list[str]:
        """
        List descriptions of all registered tools.

        Returns:
        - list[str]: List of tool descriptions.

        Example:
        ```python
        registry = SimpleToolRegistry(...)
        descriptions = registry.list_tools_descriptions()
        for desc in descriptions:
            print(desc)
        ```
        """
        LOG.warning("Function deprecated")
        return [f"{tool.name}: {tool.description}" for tool in self.tools.values()]

    def get_tools_names(self) -> list[str]:
        return [tool.name() for tool in self.tools]

    def get_tool_list(self) -> list[BaseTool]:
        LOG.warning(
            "### Warning this function has not being tested, we recommand against using it###"
        )
        # return self.tools
        return self.tools.values()

    @staticmethod
    def with_tool_modules(
        modules: list[str],
        agent: BaseAgent,
        
        memory: AbstractMemory,
        workspace: AbstractFileWorkspace,
        model_providers: dict[ModelProviderName, BaseChatModelProvider],
    ) -> "SimpleToolRegistry":
        """
        Creates and returns a new SimpleToolRegistry with tools from given modules.
        """
        # new_registry = SimpleToolRegistry(
        #     settings=SimpleToolRegistry.SystemSettings(),
        #     logger=logging.getLogger(__name__),
        #     memory=None,
        #     workspace=None,
        #     model_providers={},
        # )
        new_registry = SimpleToolRegistry(
            logger=LOG,
            settings=SimpleToolRegistry.SystemSettings(),
            memory=memory,
            workspace=workspace,
            model_providers=model_providers,
        )
        SimpleToolRegistry._agent = agent

        # LOG.trace(
        #     f"The following tool categories are disabled: {config.disabled_tool_categories}"
        # )
        # enabled_tool_modules = [
        #     x for x in modules if x not in config.disabled_tool_categories
        # ]
        enabled_tool_modules = [x for x in modules]

        LOG.trace(
            f"The following tool categories are enabled: {enabled_tool_modules}"
        )

        for tool_module in enabled_tool_modules:
            new_registry.import_tool_module(tool_module)

        # # Unregister commands that are incompatible with the current config
        # for tool in [c for c in new_registry.tools.values()]:
        #     if callable(tool.enabled) and not tool.enabled(config):
        #         new_registry.unregister(tool)
        #         LOG.trace(
        #             f"Unregistering incompatible tool '{tool.name()}': \"{tool.disabled_reason or 'Disabled by current config.'}\""
        #         )

        return new_registry

    def import_tool_module(self, module_name: str) -> None:
        """
        Imports the specified Python module containing tool plugins.

        This method imports the associated module and registers any functions or
        classes that are decorated with the `AUTO_GPT_TOOL_IDENTIFIER` attribute
        as `Tool` objects. The registered `Tool` objects are then added to the
        `tools` list of the `SimpleToolRegistry` object.

        Args:
            module_name (str): The name of the module to import for tool plugins.


        Example:
        ```python
        registry = SimpleToolRegistry(...)
        registry.import_tool_module("sample_module")
        ```
        """

        module = importlib.import_module(module_name)

        category = self.register_module_category(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            tool = None

            # Register decorated functions
            if getattr(attr, AUTO_GPT_TOOL_IDENTIFIER, False):
                tool = attr.tool

            # Register tool classes
            elif inspect.isclass(attr) and issubclass(attr, BaseTool) and attr != BaseTool:
                tool = attr()

            if tool:
                self.register(tool)
                category.tools.append(tool)

    def register_module_category(self, module: ModuleType) -> ToolCategory:
        """
        Registers a module's category in the tool registry.

        Args:
        - module (ModuleType): The module from which the category is to be registered.

        Returns:
        - ToolCategory: The registered category.

        Example:
        ```python
        registry = SimpleToolRegistry(...)
        module = ...
        category = registry.register_module_category(module)
        ```
        """
        if not (category_name := getattr(module, "TOOL_CATEGORY", None)):
            raise ValueError(f"Cannot import invalid tool module {module.__name__}")

        if category_name not in self.categories:
            self.categories[category_name] = SimpleToolRegistry.ToolCategory(
                name=category_name,
                title=getattr(
                    module, "TOOL_CATEGORY_TITLE", category_name.capitalize()
                ),
                description=getattr(module, "__doc__", ""),
            )

        category = self.categories[category_name]
        if module not in category.modules:
            category.modules.append(module)

        return category

    async def call(self, tool_name: str, **kwargs) -> ToolResult:
        LOG =  AFAASLogger(name=__name__)
        LOG.warning("ToolRegistry.call() is deprecated")

        return await self.perform(tool_name=tool_name, **kwargs)
