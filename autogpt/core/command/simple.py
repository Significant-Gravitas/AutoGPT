from autogpt.core.command.base import Command, CommandRegistry
from autogpt.core.configuration.schema import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
)


class CommandRegistryConfiguration(SystemConfiguration):
    pass


class CommandRegistrySettings(SystemSettings):
    configuration: CommandRegistryConfiguration


class SimpleCommandRegistry(CommandRegistry, Configurable):
    defaults = CommandRegistrySettings(
        name="simple_command_registry",
        description="A simple command registry.",
        configuration=CommandRegistryConfiguration(),
    )
