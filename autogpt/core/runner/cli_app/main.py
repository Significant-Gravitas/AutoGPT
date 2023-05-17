import click

from autogpt.core.agent import AgentSettings, SimpleAgent
from autogpt.core.planning import LanguageModelResponse
from autogpt.core.runner.client_lib.logging import get_client_logger


async def run_auto_gpt(user_configuration: dict):
    """Run the Auto-GPT CLI client."""

    client_logger = get_client_logger()
    client_logger.debug("Getting agent settings")

    agent_workspace = (
        user_configuration.get("workspace", {}).get("configuration", {}).get("root", "")
    )

    if not agent_workspace:  # We don't have an agent yet.
        #################
        # Bootstrapping #
        #################
        # Step 1. Collate the user's settings with the default system settings.
        agent_settings: AgentSettings = SimpleAgent.compile_settings(
            client_logger,
            user_configuration,
        )

        # Step 2. Get a name and goals for the agent.
        # First we need to figure out what the user wants to do with the agent.
        # We'll do this by asking the user for a prompt.
        user_objective = click.prompt("What do you want Auto-GPT to do?")

        # Ask a language model to determine a name and goals for a suitable agent.
        name_and_goals = await SimpleAgent.determine_agent_name_and_goals(
            user_objective,
            agent_settings,
            client_logger,
        )
        click.echo(parse_agent_name_and_goals(name_and_goals))
        # Finally, update the agent settings with the name and goals.
        agent_settings.update_agent_name_and_goals(name_and_goals.content)

        # Step 3. Provision the agent.
        agent_workspace = SimpleAgent.provision_agent(agent_settings, client_logger)
        click.echo("agent is provisioned")

    # launch agent interaction loop
    agent = SimpleAgent.from_workspace(
        agent_workspace,
        client_logger,
    )
    click.echo("agent is loaded")


def parse_agent_name_and_goals(name_and_goals: LanguageModelResponse) -> str:
    content = name_and_goals.content
    parsed_response = f"Agent Name: {content['agent_name']}\n"
    parsed_response += f"Agent Role: {content['agent_role']}\n"
    parsed_response += "Agent Goals:\n"
    for i, goal in enumerate(content["agent_goals"]):
        parsed_response += f"{i+1}. {goal}\n"
    return parsed_response
