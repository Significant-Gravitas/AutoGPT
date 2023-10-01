import logging
from logging import Logger
import importlib
import inspect
from typing import Any, Iterator
from types import ModuleType
from pydantic import BaseModel
from autogpt.core.tools.base import Tool, ToolConfiguration, ToolsRegistry
from autogpt.core.tools.builtins import BUILTIN_TOOLS
from autogpt.core.tools.schema import ToolResult
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory.base import Memory
from autogpt.core.plugin.simple import SimplePluginService
from autogpt.core.resource.model_providers import (
    BaseChatModelProvider,
    CompletionModelFunction,
    ModelProviderName,
)
from autogpt.core.workspace.base import Workspace

from autogpt.core.tools.decorators import AUTO_GPT_ABILITY_IDENTIFIER


class ToolsRegistryConfiguration(SystemConfiguration):
    """
    Configuration for the ToolRegistry subsystem.

    Attributes:
        tools: A dictionary mapping tool names to their configurations.
    """

    tools: dict[str, ToolConfiguration]


class ToolsRegistrySettings(SystemSettings):
    """
    System settings for ToolRegistry.

    Attributes:
        configuration: Configuration settings for ToolRegistry.
    """

    configuration: ToolsRegistryConfiguration


class SimpleToolRegistry(ToolsRegistry, Configurable):
    """
    A manager for a collection of Tool objects. Supports registration, modification, retrieval, and loading
    of tool plugins from a specified directory.

    Attributes:
        default_settings: Default system settings for the SimpleToolRegistry.
    """

    default_settings = ToolsRegistrySettings(
        name="simple_tool_registry",
        description="A simple tool registry.",
        configuration=ToolsRegistryConfiguration(
            tools={
                tool_name: tool.default_configuration
                for tool_name, tool in BUILTIN_TOOLS.items()
            },
        ),
    )

    class ToolCategory(BaseModel):
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
        tools: list[Tool]
        modules: list[ModuleType]

        class Config:
            arbitrary_types_allowed = True

    def __init__(
        self,
        settings: ToolsRegistrySettings,
        logger: logging.Logger,
        memory: Memory,
        workspace: Workspace,
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
        super().__init__(settings, logger)
        self._memory = memory
        self._workspace = workspace
        self._model_providers = model_providers
        self._tools: list[Tool] = []
        for (
            tool_name,
            tool_configuration,
        ) in self._configuration.tools.items():
            self.register_tool(tool_name, tool_configuration)
        self.tool_aliases = {}
        self.categories = {}

    def __contains__(self, tool_name: str):
        return tool_name in self.tools or tool_name in self.tools_aliases

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
        tool_class = SimplePluginService.get_plugin(tool_configuration.location)
        tool_args = {
            "logger": self._logger.getChild(tool_name),
            "configuration": tool_configuration,
        }
        if tool_configuration.packages_required:
            # TODO: Check packages are installed and maybe install them.
            pass
        if tool_configuration.memory_provider_required:
            tool_args["memory"] = self._memory
        if tool_configuration.workspace_required:
            tool_args["workspace"] = self._workspace
        if tool_configuration.language_model_required:
            tool_args["language_model_provider"] = self._model_providers[
                tool_configuration.language_model_required.provider_name
            ]
        tool = tool_class(**tool_args)
        self._tools.append(tool)

        for alias in aliases:
            if alias in self.tool_aliases:
                # Handle overwriting aliases or log a warning
                logging.warning(f"Alias '{alias}' is already in use.")
            else:
                self.tool_aliases[alias] = tool_name

        # Handle categorization
        if category:
            if category not in self.categories:
                self.categories[category] = self.ToolCategory(
                    name=category, title=category.capitalize(), description=""
                )
            self.categories[category].tools.append(tool_name)

    def unregister(self, tool: Tool) -> None:
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
        if tool in self._tools:
            self._tools.remove(tool)
            for alias, aliased_tool in self.tool_aliases.items():
                if aliased_tool == tool.name():
                    del self.tool_aliases[alias]
        else:
            raise KeyError(f"Tool '{tool.name()}' not found in registry.")

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
        for tool in self._tools:
            module = self._import_module(tool.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, "register_tool"):
                reloaded_tool = getattr(reloaded_module, "register_tool")
                self.register_tool(
                    reloaded_tool.name, reloaded_tool.configuration
                )

    def get_tool(self, tool_name: str) -> Tool:
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
        for tool in self._tools:
            if tool.name() == tool_name:
                return tool
        if tool_name in self.tool_aliases:
            aliased_tool_name = self.tool_aliases[tool_name]
            for tool in self._tools:
                if tool.name() == aliased_tool_name:
                    return tool
        raise ValueError(f"Tool '{tool_name}' not found.")

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
        tool = self.get_tool(tool_name)
        if tool:
            if tool.is_async == True:
                return await tool(**kwargs)
            else:
                return tool(**kwargs)

        raise KeyError(f"Tool '{tool_name}' not found in registry")

    async def call(self, tool_name: str, **kwargs) -> ToolResult:
        logger = logging.getLogger(__name__)
        logger.warning("ToolRegistry.call() is deprecated")

        return await self.perform(tool_name=tool_name, **kwargs)

    def list_available_tools(self, agent=None) -> Iterator[Tool]:
        """
        Return a generator over all available tools.

        Args:
        - agent (optional): An agent instance to check tool availtool.

        Yields:
        - Tool: Available tools.

        Example:
        ```python
        registry = SimpleToolRegistry(...)
        for tool in registry.list_available_tools():
            print(tool.name())
        ```
        """

        for tool in self._tools:
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
        # return [f"{tool.name()}: {tool.description()}" for tool in self._tools]
        return [f"{tool.name()}: {tool.description}" for tool in self._tools]

    def get_tools_names(self) -> list[str]:
        return [tool.name() for tool in self._tools]

    def get_tools(self) -> list[Tool]:
        return self._tools

    def dump_tools(self) -> list[CompletionModelFunction]:
        return [tool.spec for tool in self._tools]

    @staticmethod
    def with_tool_modules(
        modules: list[str], config: ToolsRegistryConfiguration
    ) -> "SimpleToolRegistry":
        """
        Creates and returns a new SimpleToolRegistry with tools from given modules.
        """
        logger = logging.getLogger(__name__)
        new_registry = SimpleToolRegistry(
            settings=SimpleToolRegistry.default_settings,
            logger=logging.getLogger(__name__),
            memory=None,
            workspace=None,
            model_providers={},
        )

        logger.debug(
            f"The following tool categories are disabled: {config.disabled_tool_categories}"
        )
        enabled_tool_modules = [
            x for x in modules if x not in config.disabled_tool_categories
        ]

        logger.debug(
            f"The following tool categories are enabled: {enabled_tool_modules}"
        )

        for tool_module in enabled_tool_modules:
            new_registry.import_tool_module(tool_module)

        for tool in [c for c in new_registry._tools]:
            if callable(tool.enabled) and not tool.enabled(config):
                new_registry.unregister(tool)
                logger.debug(
                    f"Unregistering incompatible tool '{tool.name()}': \"{tool.disabled_reason or 'Disabled by current config.'}\""
                )

        return new_registry

    def import_tool_module(self, module_name: str):
        """
        Imports the specified Python module containing tool plugins.

        This method imports the associated module and registers any functions or
        classes that are decorated with the `AUTO_GPT_ABILITY_IDENTIFIER` attribute
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
        module = self._import_module(module_name)
        category = self.register_module_category(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            tool = None

            # Register decorated functions
            if getattr(attr, SimpleToolRegistry.AUTO_GPT_ABILITY_IDENTIFIER, False):
                tool = attr.tool

            # Register tool classes
            elif inspect.isclass(attr) and issubclass(attr, Tool) and attr != Tool:
                tool = attr()

            if tool:
                self.register_tool(tool.name, tool.configuration)
                category.tools.append(tool.name)

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
        if not (category_name := getattr(module, "COMMAND_CATEGORY", None)):
            raise ValueError(f"Cannot import invalid tool module {module.__name__}")

        if category_name not in self.categories:
            self.categories[category_name] = self.ToolCategory(
                name=category_name,
                title=getattr(
                    module, "COMMAND_CATEGORY_TITLE", category_name.capitalize()
                ),
                description=getattr(module, "__doc__", ""),
                tools=[],
                modules=[],
            )

        category = self.categories[category_name]
        if module not in category.modules:
            category.modules.append(module)

        return category
