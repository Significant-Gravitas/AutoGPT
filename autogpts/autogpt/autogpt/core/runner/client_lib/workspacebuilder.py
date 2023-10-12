### Some of it might have to be provided by the core

import yaml
import click
from pathlib import Path
from logging import Logger
from autogpt.core.agents import (
    PlannerAgent,
)  ### TODO should work for every Agent
from autogpt.core.runner.client_lib.logging import get_client_logger
from autogpt.core.runner.client_lib.parser import (
    parse_agent_name_and_goals,
    parse_ability_result,
    parse_agent_plan,
    parse_next_ability,
)

DEFAULT_SETTINGS_FILE = str(Path("~/auto-gpt/default_agent_settings.yml").expanduser())


async def workspace_loader(
    user_configuration: dict, client_logger: Logger, agent_workspace
):
    """Run the Auto-GPT CLI client."""

    # Step 1. Collate the user's settings with the default system settings.
    agent_settings: PlannerAgent.SystemSettings = PlannerAgent.SystemSettings()

    # Step 2. Get a name and goals for the agent.
    # First we need to figure out what the user wants to do with the agent.
    # We'll do this by asking the user for a prompt.

    user_objective = click.prompt("What do you want Auto-GPT to do?")
    # Ask a language model to determine a name and goals for a suitable agent.
    name_and_goals = await PlannerAgent.determine_agent_name_and_goals(
        user_objective,
        agent_settings,
        client_logger,
    )

    # parsed_agent_goals = parse_agent_name_and_goals(name_and_goals)
    # print(parsed_agent_goals)
    # # Finally, update the agent settings with the name and goals.
    # agent_settings.update_agent_name_and_goals(name_and_goals)

    # Step 3. Provision the agent.
    agent_workspace = PlannerAgent.provision_agent(agent_settings, client_logger)
    print("agent is provisioned")
    return agent_workspace


def get_logger_and_workspace(user_configuration: dict):
    client_logger = get_client_logger()
    client_logger.debug("Getting agent settings")

    agent_workspace = (
        user_configuration.get("workspace", {}).get("configuration", {}).get("root", "")
    )
    return client_logger, agent_workspace


def get_settings_from_file():
    """
    Current Back-end for setting is a File
    """
    # TODO : Possible Back-End No SQL Database
    settings_file = Path(DEFAULT_SETTINGS_FILE)
    settings = {}
    if settings_file.exists():
        settings = yaml.safe_load(settings_file.read_text())
    return settings
