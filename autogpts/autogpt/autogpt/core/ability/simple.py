import logging
from logging import Logger
import importlib
import inspect
from typing import Any, Iterator 
from types import ModuleType
from pydantic import BaseModel
from autogpt.core.ability.base import Ability, AbilityConfiguration, AbilityRegistry
from autogpt.core.ability.builtins import BUILTIN_ABILITIES
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory.base import Memory
from autogpt.core.plugin.simple import SimplePluginService
from autogpt.core.resource.model_providers import (
    ChatModelProvider,
    CompletionModelFunction,
    ModelProviderName,
)
from autogpt.core.workspace.base import Workspace

from autogpt.core.ability.decorators import AUTO_GPT_ABILITY_IDENTIFIER


class AbilityRegistryConfiguration(SystemConfiguration):
    """
    Configuration for the AbilityRegistry subsystem.

    Attributes:
        abilities: A dictionary mapping ability names to their configurations.
    """
    abilities: dict[str, AbilityConfiguration]


class AbilityRegistrySettings(SystemSettings):
    """
    System settings for AbilityRegistry.

    Attributes:
        configuration: Configuration settings for AbilityRegistry.
    """
    configuration: AbilityRegistryConfiguration

class SimpleAbilityRegistry(AbilityRegistry, Configurable):
    """
    A manager for a collection of Ability objects. Supports registration, modification, retrieval, and loading
    of ability plugins from a specified directory.

    Attributes:
        default_settings: Default system settings for the SimpleAbilityRegistry.
    """

    default_settings = AbilityRegistrySettings(
        name="simple_ability_registry",
        description="A simple ability registry.",
        configuration=AbilityRegistryConfiguration(
            abilities={
                ability_name: ability.default_configuration
                for ability_name, ability in BUILTIN_ABILITIES.items()
            },
        ),
    )

    class AbilityCategory(BaseModel):
        """
        Represents a category of abilities.

        Attributes:
            name: Name of the category.
            title: Display title for the category.
            description: Description of the category.
            abilities: List of Ability objects associated with the category.
            modules: List of ModuleType objects related to the category.
        """
        name: str
        title: str
        description: str
        abilities: list[Ability]
        modules: list[ModuleType]

        class Config:
            arbitrary_types_allowed = True

    def __init__(
        self,
        settings: AbilityRegistrySettings,
        logger: logging.Logger,
        memory: Memory,
        workspace: Workspace,
        model_providers: dict[ModelProviderName, ChatModelProvider],
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
        self._abilities : list[Ability]= []
        for (
            ability_name,
            ability_configuration,
        ) in self._configuration.abilities.items():
            self.register_ability(ability_name, ability_configuration)
        self.ability_aliases = {}
        self.categories = {}

    def __contains__(self, ability_name: str):
        return ability_name in self.abilities or ability_name in self.abilities_aliases

    def _import_module(self, module_name: str):
        """Imports a module using its name."""
        return importlib.import_module(module_name)

    def _reload_module(self, module):
        """Reloads a given module."""
        return importlib.reload(module)

    def register_ability(
        self, ability_name: str, ability_configuration: AbilityConfiguration, aliases: list[str] = [], category: str = None
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
        self._abilities.append(ability)

        for alias in aliases:
            if alias in self.ability_aliases:
                # Handle overwriting aliases or log a warning
                logging.warning(f"Alias '{alias}' is already in use.")
            else:
                self.ability_aliases[alias] = ability_name

        # Handle categorization
        if category:
            if category not in self.categories:
                self.categories[category] = self.AbilityCategory(name=category, title=category.capitalize(), description="")
            self.categories[category].abilities.append(ability_name)

   
    def unregister(self, ability: Ability) -> None:
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
        if ability in self._abilities:
            self._abilities.remove(ability)
            for alias, aliased_ability in self.ability_aliases.items():
                if aliased_ability == ability.name():
                    del self.ability_aliases[alias]
        else:
            raise KeyError(f"Ability '{ability.name()}' not found in registry.")

    def reload_abilities(self) -> None:
        """
        Reloads all loaded ability plugins.

        Useful for dynamically updating abilities without restarting the application.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        registry.reload_abilities()
        ```
        """
        for ability in self._abilities:
            module = self._import_module(ability.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, "register_ability"):
                reloaded_ability = getattr(reloaded_module, "register_ability")
                self.register_ability(reloaded_ability.name, reloaded_ability.configuration)

 
    def get_ability(self, ability_name: str) -> Ability:
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
        for ability in self._abilities:
            if ability.name() == ability_name:
                return ability
        if ability_name in self.ability_aliases:
            aliased_ability_name = self.ability_aliases[ability_name]
            for ability in self._abilities:
                if ability.name() == aliased_ability_name:
                    return ability
        raise ValueError(f"Ability '{ability_name}' not found.")

    async def perform(self, ability_name: str, **kwargs) -> AbilityResult:
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
            if ability.is_async == True : 
                return await ability(**kwargs)
            else :
                return  ability(**kwargs)
        
        raise KeyError(f"Ability '{ability_name}' not found in registry")
    
    async def call(self, ability_name: str, **kwargs) -> AbilityResult:
        logger = logging.getLogger(__name__)
        logger.warning("AbilityRegistry.call() is deprecated")

        return await self.perform(ability_name=ability_name , **kwargs)
    
    def list_available_abilities(self, agent=None) -> Iterator[Ability]:
        """
        Return a generator over all available abilities.

        Args:
        - agent (optional): An agent instance to check ability availability.

        Yields:
        - Ability: Available abilities.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        for ability in registry.list_available_abilities():
            print(ability.name())
        ```
        """

        for ability in self._abilities:
            available = ability.available
            if callable(ability.available):
                available = ability.available(agent)
            if available:
                yield ability

    def list_abilities_descriptions(self) -> list[str]:
        """
        List descriptions of all registered abilities.

        Returns:
        - list[str]: List of ability descriptions.

        Example:
        ```python
        registry = SimpleAbilityRegistry(...)
        descriptions = registry.list_abilities_descriptions()
        for desc in descriptions:
            print(desc)
        ```
        """
        return [
            f"{ability.name()}: {ability.description()}" for ability in self._abilities
        ]

    def get_abilities_names(self) -> list[str]:
        return [
            ability.name() for ability in self._abilities
        ]

    def get_abilities(self) -> list[Ability]:
        return self._abilities

    def dump_abilities(self) -> list[dict]:
        return [ability.dump() for ability in self._abilities]


    @staticmethod
    def with_ability_modules(modules: list[str], config: AbilityRegistryConfiguration) -> "SimpleAbilityRegistry":
        """
        Creates and returns a new SimpleAbilityRegistry with abilities from given modules.
        """
        logger = logging.getLogger(__name__)
        new_registry = SimpleAbilityRegistry(
            settings=SimpleAbilityRegistry.default_settings,
            logger=logging.getLogger(__name__),
            memory=None, 
            workspace=None, 
            model_providers={} 
        )

        logger.debug(f"The following ability categories are disabled: {config.disabled_ability_categories}")
        enabled_ability_modules = [
            x for x in modules if x not in config.disabled_ability_categories
        ]

        logger.debug(f"The following ability categories are enabled: {enabled_ability_modules}")

        for ability_module in enabled_ability_modules:
            new_registry.import_ability_module(ability_module)

        for ability in [c for c in new_registry._abilities]:
            if callable(ability.enabled) and not ability.enabled(config):
                new_registry.unregister(ability)
                logger.debug(f"Unregistering incompatible ability '{ability.name()}': \"{ability.disabled_reason or 'Disabled by current config.'}\"")

        return new_registry
    
    def import_ability_module(self, module_name: str):
        """
        Imports the specified Python module containing ability plugins.

        This method imports the associated module and registers any functions or
        classes that are decorated with the `AUTO_GPT_ABILITY_IDENTIFIER` attribute
        as `Ability` objects. The registered `Ability` objects are then added to the
        `abilities` list of the `SimpleAbilityRegistry` object.

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
            elif inspect.isclass(attr) and issubclass(attr, Ability) and attr != Ability:
                ability = attr()

            if ability:
                self.register_ability(ability.name, ability.configuration)
                category.abilities.append(ability.name)

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
                title=getattr(module, "COMMAND_CATEGORY_TITLE", category_name.capitalize()),
                description=getattr(module, "__doc__", ""),
                abilities=[],
                modules=[]
            )

        category = self.categories[category_name]
        if module not in category.modules:
            category.modules.append(module)

        return category