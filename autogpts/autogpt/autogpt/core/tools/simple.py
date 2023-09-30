import logging
from logging import Logger
import importlib
import inspect
from typing import Any, Iterator
from types import ModuleType
from pydantic import BaseModel
from autogpt.core.tools.base import Tool, ToolConfiguration, ToolsRegistry
from autogpt.core.tools.builtins import BUILTIN_ABILITIES
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
    Configuration for the AbilityRegistry subsystem.

    Attributes:
        tools: A dictionary mapping ability names to their configurations.
    """

    tools: dict[str, ToolConfiguration]


class ToolsRegistrySettings(SystemSettings):
    """
    System settings for AbilityRegistry.

    Attributes:
        configuration: Configuration settings for AbilityRegistry.
    """

    configuration: ToolsRegistryConfiguration


class SimpleAbilityRegistry(ToolsRegistry, Configurable):
    """
    A manager for a collection of Ability objects. Supports registration, modification, retrieval, and loading
    of ability plugins from a specified directory.

    Attributes:
        default_settings: Default system settings for the SimpleAbilityRegistry.
    """

    default_settings = ToolsRegistrySettings(
        name="simple_tool_registry",
        description="A simple ability registry.",
        configuration=ToolsRegistryConfiguration(
            tools={
                ability_name: ability.default_configuration
                for ability_name, ability in BUILTIN_ABILITIES.items()
            },
        ),
    )

    class AbilityCategory(BaseModel):
        """
        Represents a category of tools.

        Attributes:
            name: Name of the category.
            title: Display title for the category.
            description: Description of the category.
            tools: List of Ability objects associated with the category.
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
        Initialize the SimpleAbilityRegistry.

        Args:
            settings: Configuration settings for the registry.
            logger: Logging instance to use for the registry.
            memory: Memory instance for the registry.
            workspace: Workspace instance for the registry.
            model_providers: A dictionary mapping model provider names to chat model providers.

        Example:
            registry = SimpleAbilityRegistry(settings, logger, memory, workspace, model_providers)
        """
        super().__init__(settings, logger)
        self._memory = memory
        self._workspace = workspace
        self._model_providers = model_providers
        self._tools: list[Tool] = []
        for (
            ability_name,
            ability_configuration,
        ) in self._configuration.tools.items():
            self.register_ability(ability_name, ability_configuration)
        self.ability_aliases = {}
        self.categories = {}

    def __contains__(self, ability_name: str):
        return ability_name in self.tools or ability_name in self.tools_aliases

    def _import_module(self, module_name: str):
        """Imports a module using its name."""
        return importlib.import_module(module_name)

    def _reload_module(self, module):
        """Reloads a given module."""
        return importlib.reload(module)

    def register_ability(
        self,
        ability_name: str,
        ability_configuration: ToolConfiguration,
        aliases: list[str] = [],
        category: str = None,
    ) -> None:
        """
        Register a new ability with the registry.

        Args:
        - ability_name (str): Name of the ability.
        - ability_configuration (AbilityConfiguration): Configuration details for the ability.
        - aliases (list[str], optional): A list of alternative names for the ability. Defaults to an empty list.
        - category (str, optional): Category to which the ability belongs. Defaults to None.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        registry.register_ability("sample_ability", AbilityConfiguration(), aliases=["example"], category="sample_category")
        ```
        """
        ability_class = SimplePluginService.get_plugin(ability_configuration.location)
        ability_args = {
            "logger": self._logger.getChild(ability_name),
            "configuration": ability_configuration,
        }
        if ability_configuration.packages_required:
            # TODO: Check packages are installed and maybe install them.
            pass
        if ability_configuration.memory_provider_required:
            ability_args["memory"] = self._memory
        if ability_configuration.workspace_required:
            ability_args["workspace"] = self._workspace
        if ability_configuration.language_model_required:
            ability_args["language_model_provider"] = self._model_providers[
                ability_configuration.language_model_required.provider_name
            ]
        ability = ability_class(**ability_args)
        self._tools.append(ability)

        for alias in aliases:
            if alias in self.ability_aliases:
                # Handle overwriting aliases or log a warning
                logging.warning(f"Alias '{alias}' is already in use.")
            else:
                self.ability_aliases[alias] = ability_name

        # Handle categorization
        if category:
            if category not in self.categories:
                self.categories[category] = self.AbilityCategory(
                    name=category, title=category.capitalize(), description=""
                )
            self.categories[category].tools.append(ability_name)

    def unregister(self, ability: Tool) -> None:
        """
        Unregister an ability from the registry.

        Args:
        - ability (Ability): The ability instance to unregister.

        Raises:
        - KeyError: If the ability is not found in the registry.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        ability_instance = ...
        registry.unregister(ability_instance)
        ```
        """
        if ability in self._tools:
            self._tools.remove(ability)
            for alias, aliased_ability in self.ability_aliases.items():
                if aliased_ability == ability.name():
                    del self.ability_aliases[alias]
        else:
            raise KeyError(f"Ability '{ability.name()}' not found in registry.")

    def reload_tools(self) -> None:
        """
        Reloads all loaded ability plugins.

        Useful for dynamically updating tools without restarting the application.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        registry.reload_tools()
        ```
        """
        for ability in self._tools:
            module = self._import_module(ability.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, "register_ability"):
                reloaded_ability = getattr(reloaded_module, "register_ability")
                self.register_ability(
                    reloaded_ability.name, reloaded_ability.configuration
                )

    def get_ability(self, ability_name: str) -> Tool:
        """
        Retrieve a specific ability by its name.

        Args:
        - ability_name (str): Name of the ability to retrieve.

        Returns:
        - Ability: The matched ability.

        Raises:
        - ValueError: If the ability is not found.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        ability = registry.get_ability("sample_ability")
        ```
        """
        for ability in self._tools:
            if ability.name() == ability_name:
                return ability
        if ability_name in self.ability_aliases:
            aliased_ability_name = self.ability_aliases[ability_name]
            for ability in self._tools:
                if ability.name() == aliased_ability_name:
                    return ability
        raise ValueError(f"Ability '{ability_name}' not found.")

    async def perform(self, ability_name: str, **kwargs) -> ToolResult:
        """
        Retrieve a registered ability by its name.

        Args:
            ability_name: Name or alias of the ability to retrieve.

        Returns:
            The requested Ability instance.

        Raises:
            ValueError: If no ability with the given name or alias is found.

        Example:
            ability = registry.get_ability("example_ability")
        """
        ability = self.get_ability(ability_name)
        if ability:
            if ability.is_async == True:
                return await ability(**kwargs)
            else:
                return ability(**kwargs)

        raise KeyError(f"Ability '{ability_name}' not found in registry")

    async def call(self, ability_name: str, **kwargs) -> ToolResult:
        logger = logging.getLogger(__name__)
        logger.warning("AbilityRegistry.call() is deprecated")

        return await self.perform(ability_name=ability_name, **kwargs)

    def list_available_tools(self, agent=None) -> Iterator[Tool]:
        """
        Return a generator over all available tools.

        Args:
        - agent (optional): An agent instance to check ability availability.

        Yields:
        - Ability: Available tools.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        for ability in registry.list_available_tools():
            print(ability.name())
        ```
        """

        for ability in self._tools:
            available = ability.available
            if callable(ability.available):
                available = ability.available(agent)
            if available:
                yield ability

    def list_tools_descriptions(self) -> list[str]:
        """
        List descriptions of all registered tools.

        Returns:
        - list[str]: List of ability descriptions.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        descriptions = registry.list_tools_descriptions()
        for desc in descriptions:
            print(desc)
        ```
        """
        return [f"{ability.name()}: {ability.description()}" for ability in self._tools]

    def get_tools_names(self) -> list[str]:
        return [ability.name() for ability in self._tools]

    def get_tools(self) -> list[Tool]:
        return self._tools

    def dump_tools(self) -> list[dict]:
        return [ability.dump() for ability in self._tools]

    @staticmethod
    def with_ability_modules(
        modules: list[str], config: ToolsRegistryConfiguration
    ) -> "SimpleAbilityRegistry":
        """
        Creates and returns a new SimpleAbilityRegistry with tools from given modules.
        """
        logger = logging.getLogger(__name__)
        new_registry = SimpleAbilityRegistry(
            settings=SimpleAbilityRegistry.default_settings,
            logger=logging.getLogger(__name__),
            memory=None,
            workspace=None,
            model_providers={},
        )

        logger.debug(
            f"The following ability categories are disabled: {config.disabled_ability_categories}"
        )
        enabled_ability_modules = [
            x for x in modules if x not in config.disabled_ability_categories
        ]

        logger.debug(
            f"The following ability categories are enabled: {enabled_ability_modules}"
        )

        for ability_module in enabled_ability_modules:
            new_registry.import_ability_module(ability_module)

        for ability in [c for c in new_registry._tools]:
            if callable(ability.enabled) and not ability.enabled(config):
                new_registry.unregister(ability)
                logger.debug(
                    f"Unregistering incompatible ability '{ability.name()}': \"{ability.disabled_reason or 'Disabled by current config.'}\""
                )

        return new_registry

    def import_ability_module(self, module_name: str):
        """
        Imports the specified Python module containing ability plugins.

        This method imports the associated module and registers any functions or
        classes that are decorated with the `AUTO_GPT_ABILITY_IDENTIFIER` attribute
        as `Ability` objects. The registered `Ability` objects are then added to the
        `tools` list of the `SimpleAbilityRegistry` object.

        Args:
            module_name (str): The name of the module to import for ability plugins.


        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        registry.import_ability_module("sample_module")
        ```
        """
        module = self._import_module(module_name)
        category = self.register_module_category(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            ability = None

            # Register decorated functions
            if getattr(attr, SimpleAbilityRegistry.AUTO_GPT_ABILITY_IDENTIFIER, False):
                ability = attr.ability

            # Register ability classes
            elif inspect.isclass(attr) and issubclass(attr, Tool) and attr != Tool:
                ability = attr()

            if ability:
                self.register_ability(ability.name, ability.configuration)
                category.tools.append(ability.name)

    def register_module_category(self, module: ModuleType) -> AbilityCategory:
        """
        Registers a module's category in the ability registry.

        Args:
        - module (ModuleType): The module from which the category is to be registered.

        Returns:
        - AbilityCategory: The registered category.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        module = ...
        category = registry.register_module_category(module)
        ```
        """
        if not (category_name := getattr(module, "COMMAND_CATEGORY", None)):
            raise ValueError(f"Cannot import invalid ability module {module.__name__}")

        if category_name not in self.categories:
            self.categories[category_name] = self.AbilityCategory(
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
