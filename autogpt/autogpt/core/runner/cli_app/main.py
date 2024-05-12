import click

from autogpt.core.agent import AgentSettings, SimpleAgent
from autogpt.core.runner.client_lib.logging import (
    configure_root_logger,
    get_client_logger,
)
from autogpt.core.runner.client_lib.parser import (
    parse_ability_result,
    parse_agent_name_and_goals,
    parse_agent_plan,
    parse_next_ability,
)


async def run_auto_gpt(user_configuration: dict):
    """Run the AutoGPT CLI client."""

    configure_root_logger()

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
        user_objective = click.prompt("What do you want AutoGPT to do?")
        # Ask a language model to determine a name and goals for a suitable agent.
        name_and_goals = await SimpleAgent.determine_agent_name_and_goals(
            user_objective,
            agent_settings,
            client_logger,
        )
        print("\n" + parse_agent_name_and_goals(name_and_goals))
        # Finally, update the agent settings with the name and goals.
        agent_settings.update_agent_name_and_goals(name_and_goals)

        # Step 3. Provision the agent.
        agent_workspace = SimpleAgent.provision_agent(agent_settings, client_logger)
        client_logger.info("Agent is provisioned")

    # launch agent interaction loop
    agent = SimpleAgent.from_workspace(
        agent_workspace,
        client_logger,
    )
    client_logger.info("Agent is loaded")

    plan = await agent.build_initial_plan()
    print(parse_agent_plan(plan))

    while True:
        current_task, next_ability = await agent.determine_next_ability(plan)
        print(parse_next_ability(current_task, next_ability))
        user_input = click.prompt(
            "Should the agent proceed with this ability?",
            default="y",
        )
        ability_result = await agent.execute_next_ability(user_input)
        print(parse_ability_result(ability_result))
