import logging

import click

from AFAAS.core.lib.sdk.logger import AFAASLogger
from AFAAS.core.agents import PlannerAgent


async def handle_user_input_request(prompt):
    user_input = click.prompt(
        prompt,
        default="y",
    )
    return user_input


async def handle_user_message(prompt):
    print(prompt)


async def run_cli_demo():
    """Run the AFAAS CLI (Command Line Interface) Demonstration"""

    LOG = AFAASLogger(name = __name__)
    LOG.info("Getting agent settings")

    import uuid

    #
    # Step 1. Get the user
    # 
    #
    LOG.notice("AFAAS Data Structure support multiple users (however since there is no UI to enforce that we will be using a user with ID : a1621e69-970a-4340-86e7-778d82e2137b")
    user_id: str = "A" + str(uuid.UUID("a1621e69-970a-4340-86e7-778d82e2137b"))
    agent_settings: PlannerAgent.SystemSettings = PlannerAgent.SystemSettings(
        user_id=user_id
    )

    # NOTE : Real world scenario, this user_id will be passed as an argument
    agent_dict_list: list[
        PlannerAgent.SystemSettings
    ] = PlannerAgent.list_users_agents_from_memory(
        user_id=user_id
    )

    if len(agent_dict_list) > 0 : 
        LOG.info(f"User {user_id} has {len(agent_dict_list)} agents.")
        if  LOG.level >= logging.DEBUG:
            print("This is the agents thathave been saved :")
            for i, agent_dict in enumerate(agent_dict_list):
                print(f"{i+1}. {agent_dict.agent_name}({agent_dict.agent_id}) : {agent_dict.agent_goal_sentence}")

            selected_agent_index : int = 0 
            i : int = 0
            while selected_agent_index < 1 or selected_agent_index > len(agent_dict_list):
                if i > 0 :
                    print("Ooops ! The selected agent is not in the list. Please select a valid agent.")
                i += 1
                selected_agent_index = input("Select an agent to load (Press \"C\" to create a new agent) : ")
                if selected_agent_index.lower() == "c":
                    selected_agent_index = 0
                    break

                if selected_agent_index.isdigit():
                    selected_agent_index = int(selected_agent_index)
        else :
            selected_agent_index : int = 1 
        
        agent_id = agent_dict_list[selected_agent_index - 1].agent_id
        LOG.info(
            f"Loading agent {agent_id} from get_agentsetting_list_from_memory"
        )
        agent : PlannerAgent = PlannerAgent.get_instance_from_settings(
            agent_settings=agent_dict_list[selected_agent_index - 1],
        )

        # agent_from_memory = None
        # LOG.info(f"Loading agent {agent_id} from get_agent_from_memory")
        # agent_from_memory: PlannerAgent = PlannerAgent.get_agent_from_memory(
        #     agent_settings=agent_settings,
        #     agent_id=agent_id,
        #     user_id=user_id,
        #     logger=LOG,
        #     )

    else :
        #
        # New requirement gathering process
        #
        if LOG.level <= logging.DEBUG :
            user_objective = (
                "Provide a step-by-step guide on how to build a Pizza oven."
            )
        else:
            user_objective = handle_user_input_request(
                "What do you want Auto-GPT to do? (We will make Pancakes for our tests...)"
            )

        agent_settings.agent_goal_sentence = user_objective

        # agent_settings.agent_class = "PlannerAgent"
        agent_settings._type_ = "AFAAS.core.agents.planner.main.PlannerAgent"
        # agent_settings.load_root_values()

        # Step 3. Create the agent.
        agent: PlannerAgent = PlannerAgent.create_agent(
            agent_settings=agent_settings,
            LOG=LOG,
        )

    await agent.run(
        user_input_handler=handle_user_input_request,
        user_message_handler=handle_user_message,
        goal=None,
    )
