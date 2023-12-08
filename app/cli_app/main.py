import logging

import click

from AFAAS.app.sdk import forge_log
from AFAAS.app.core.agents import PlannerAgent
from app.client_lib.logging import \
    get_client_logger


async def handle_user_input_request(prompt):
    user_input = click.prompt(
        prompt,
        default="y",
    )
    return user_input


async def handle_user_message(prompt):
    print(prompt)


async def run_auto_gpt():
    """Run the Auto-GPT CLI client."""

    DEMO = True
    # INFO = 20
    # DEBUG = 10
    # NOTSET = 0
    client_logger = forge_log.ForgeLogger(__name__)
    client_logger.info("Getting agent settings")

    import uuid

    #
    # We support multiple users however since there is no UI to enforce that we will be using a user with ID : a1621e69-970a-4340-86e7-778d82e2137b
    #
    user_id: str = "A" + str(uuid.UUID("a1621e69-970a-4340-86e7-778d82e2137b"))
    # Step 1. Collate the user's settings with the default system settings.
    agent_settings: PlannerAgent.SystemSettings = PlannerAgent.SystemSettings(
        user_id=user_id
    )

    # NOTE : Real world scenario, this user_id will be passed as an argument
    agent_dict_list: list[
        PlannerAgent.SystemSettings
    ] = PlannerAgent.list_users_agents_from_memory(
        user_id=user_id, logger=client_logger
    )

    agent_from_list = None
    agent_from_memory = None
    # NOTE : This is a demonstration
    # In our demonstration we will instanciate the first agent of a given user if it exists
    if agent_dict_list:
        client_logger.info(f"User {user_id} has {len(agent_dict_list)} agents.")
        agent_id = agent_dict_list[0].agent_id

        client_logger.info(
            f"Loading agent {agent_id} from get_agentsetting_list_from_memory"
        )
        # agent_settings.update_agent_name_and_goals(agent_dict_list[0])
        agent_from_list: PlannerAgent = PlannerAgent.get_instance_from_settings(
            agent_settings=agent_dict_list[0],
            logger=client_logger,
        )

        client_logger.info(f"Loading agent {agent_id} from get_agent_from_memory")
        agent_from_memory: PlannerAgent = PlannerAgent.get_agent_from_memory(
            agent_settings=agent_settings,
            agent_id=agent_id,
            user_id=user_id,
            logger=client_logger,
        )

        client_logger.info(
            f"Comparing agents from agent list and get_agent_from_memory"
        )

        if client_logger.level == logging.DEBUG:
            if str(agent_from_memory._configuration) == str(
                agent_from_list._configuration
            ):
                client_logger.debug(
                    f"Agents from agent list and get_agent_from_memory are equal"
                )
            else:
                client_logger.debug(
                    f"Agents from agent list and get_agent_from_memory are different"
                )
                client_logger.debug(
                    f"Agents from agent list : {agent_from_list.agent_id}"
                )
                client_logger.debug(
                    f"Agents from get_agent_from_memory : {agent_from_memory.agent_id}"
                )

    # NOTE : We continue our tests with the agent from the memory as it more realistic
    agent = agent_from_memory

    if not agent:  # We don't have an agent matching this ID
        # # Step 2. Get a name and goals for the agent.
        # # First we need to figure out what the user wants to do with the agent.
        # if DEMO:
        #     # We'll use a default objective for the demo.
        #     name_and_goals = {'agent_name': 'FactoryBuilderTest',
        #                       'agent_role': 'An automated engineering expert AI, specializing in settling factories in new countries.',
        #                       'agent_goals': ['Provide a step-by-step guide on how to build a plant.',
        #                                       'Ensure compliance with local regulation',
        #                                       'Identification & selection of partners.',
        #                                       'Follow implementation untill completion',
        #                                         'Ensure realization meet quality standards.'],
        #                                         }
        # else :
        #     # We'll do this by asking the user for a prompt.
        #     user_objective = click.prompt("What do you want Auto-GPT to do? (We will make Pancakes for our tests...)")
        #     # Ask a language model to determine a name and goals for a suitable agent.
        #     name_and_goals = await PlannerAgent.determine_agent_name_and_goals(
        #         user_objective,
        #         agent_settings,
        #         client_logger,
        #     )

        # # Finally, update the agent settings with the name and goals.
        # agent_settings.update_agent_name_and_goals(name_and_goals)

        #
        # New requirement gathering process
        #
        if DEMO:
            user_objective = (
                "Provide a step-by-step guide on how to build a Pizza oven."
            )
        else:
            user_objective = handle_user_input_request(
                "What do you want Auto-GPT to do? (We will make Pancakes for our tests...)"
            )

        agent_settings.agent_goal_sentence = user_objective

        # agent_settings.agent_class = "PlannerAgent"
        agent_settings._type_ = "AFAAS.app.core.agents.planner.main.PlannerAgent"
        # agent_settings.load_root_values()

        # Step 3. Create the agent.
        agent: PlannerAgent = PlannerAgent.create_agent(
            agent_settings=agent_settings,
            logger=client_logger,
        )

    await agent.run(
        user_input_handler=handle_user_input_request,
        user_message_handler=handle_user_message,
        goal=None,
    )
