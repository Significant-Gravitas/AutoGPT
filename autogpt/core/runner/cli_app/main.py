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
        print(parse_agent_name_and_goals(name_and_goals))
        # Finally, update the agent settings with the name and goals.
        agent_settings.update_agent_name_and_goals(name_and_goals)

        # Step 3. Provision the agent.
        agent_workspace = SimpleAgent.provision_agent(agent_settings, client_logger)
        print("agent is provisioned")

    # launch agent interaction loop
    agent = SimpleAgent.from_workspace(
        agent_workspace,
        client_logger,
    )
    print("agent is loaded")

    plan = await agent.build_initial_plan()
    print(parse_agent_plan(plan))

    user_input = ""
    while True:
        agent_response = await agent.step(user_input)
        print(agent_response.content["text"])
        user_input = input("What do you want to say to the agent?")


# FIXME: We shouldn't be getting raw responses from the language model to parse here.  TBD on format.

def parse_agent_name_and_goals(name_and_goals: dict) -> str:
    parsed_response = f"Agent Name: {name_and_goals['agent_name']}\n"
    parsed_response += f"Agent Role: {name_and_goals['agent_role']}\n"
    parsed_response += "Agent Goals:\n"
    for i, goal in enumerate(name_and_goals["agent_goals"]):
        parsed_response += f"{i+1}. {goal}\n"
    return parsed_response


def parse_agent_plan(plan: dict) -> str:
    parsed_response = f"Agent Plan:\n"
    for i, task in enumerate(plan['task_list']):
        parsed_response += f"{i+1}. {task['objective']}\n"
        parsed_response += f"Task type: {task['task_type']}  "
        parsed_response += f"Priority: {task['priority']}\n"
        parsed_response += f"Ready Criteria:\n"
        for j, criteria in enumerate(task['ready_criteria']):
            parsed_response += f"    {j+1}. {criteria}\n"
        parsed_response += f"Acceptance Criteria:\n"
        for j, criteria in enumerate(task['acceptance_criteria']):
            parsed_response += f"    {j+1}. {criteria}\n"
        parsed_response += "\n"

    return parsed_response
