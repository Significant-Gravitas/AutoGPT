#!/usr/bin/env python3
import json
import logging
from argparse import ArgumentParser
from uuid import uuid4

from aiohttp import web
from aiohttp.web import Request, Response
from colorama import Fore

from autogpt.agent import Agent
from autogpt.app import execute_command
from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.logs import logger
from autogpt.memory import get_memory

parser = ArgumentParser()
parser.add_argument("port", type=int, nargs="?", default=8888)
args = parser.parse_args()

agents: dict[str, Agent] = {}


async def create_agent(request: Request) -> Response:
    config: dict = await request.json()
    agent_id = str(uuid4())

    ai_name = config["ai_name"]
    ai_role = config["ai_role"]
    ai_goals = config["ai_goals"]
    ai_config = AIConfig(ai_name, ai_role, ai_goals)

    cfg = Config()
    system_prompt = ai_config.construct_full_prompt()
    full_message_history = []
    next_action_count = 0
    triggering_prompt = "Determine which next command to use, and respond using the format specified above:"
    memory = get_memory(cfg, init=True)
    logger.typewriter_log(
        f"Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
    )
    logger.typewriter_log(f"Using Browser:", Fore.GREEN, cfg.selenium_web_browser)
    agent = Agent(
        ai_name=ai_config.ai_name,
        memory=memory,
        full_message_history=full_message_history,
        next_action_count=next_action_count,
        system_prompt=system_prompt,
        triggering_prompt=triggering_prompt,
    )
    agents[agent_id] = agent

    return Response(text=json.dumps({"agent_id": agent_id}))


async def invoke_agent(request: Request) -> Response:
    config = await request.json()
    agent_id = config["agent_id"]
    agent = agents[agent_id]

    agent.start_interaction_loop()
    user, assistant, system = agent.full_message_history[-3:]

    return Response(
        text=json.dumps(
            {
                "user": user,
                "assistant": assistant,
                "system": system,
            }
        )
    )


async def delete_agent(request: Request) -> Response:
    config = await request.json()
    agent_id = config["agent_id"]
    del agents[agent_id]
    return Response(status=204)


app = web.Application()
app.router.add_post("/create_agent", create_agent)
app.router.add_post("/invoke_agent", invoke_agent)
app.router.add_post("/delete_agent", delete_agent)

if __name__ == "__main__":
    cfg = Config()
    cfg.set_continuous_mode(True)
    cfg.set_continuous_limit(1)
    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
    web.run_app(app, port=args.port)
