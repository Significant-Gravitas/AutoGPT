"""Commands to interact with the user"""

from __future__ import annotations
import urllib
import asyncio
import requests
import pprint
#from autogpt.models.agent_actions import (    Action)
import json
COMMAND_CATEGORY = "user_interaction"
COMMAND_CATEGORY_TITLE = "User Interaction"

from autogpt.agents.agent import Agent
from autogpt.app.utils import clean_input
from autogpt.command_decorator import command
from autogpt.models.action_history import     Action
from autogpt.core.utils.json_schema import JSONSchema

#import  ai_ticket.events.inference
import ai_ticket.backends.pygithub

@command(
    "ask_user",
    (
        "If you need more details or information regarding the given goals,"
        " you can ask the user for input"
    ),
    {
        "question": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The question or prompt to the user",
            required=True,
        )
    },
    enabled=True,
)
async def ask_user(question: str, agent: Agent) -> str:
    resp = await clean_input(
        agent.legacy_config, f"{agent.ai_config.ai_name} asks: '{question}': "
    )
    return f"The user's answer: '{resp}'"


@command(
    "request_assistance",
    (
        "If you have raised a ticket and need help with it,"

    ),
    {
        "ticket_url": {
            "type": "string",
            "description": "The ticket url",
            "required": True,
        }
    },
    enabled=True,
)
async def request_assistence(ticket_url: str, next_action: str, agent: Agent) -> str:
    print("starti",ticket_url)
    #raise Exception ("testo")
    #config.github_api_key
    issue_last_comment = ""
    # try:
    response = requests.get(ticket_url)
    data = response.json()
    issue_last_comment = data

    #body = data["body"][3:-3] #```AAA```

    class Foo :        
        response = {
            "content" :data["body"]    
        }
    agent.event_history.rewind(1)
    data1 = agent.parse_and_process_response(Foo())

    next_command_name, next_command_args, assistant_reply_dict = data1
    result = "no data"
    try:
        agent.event_history.rewind(1)

        act = Action(
            name=next_command_name,
            args=next_command_args,
            reasoning=str(assistant_reply_dict),
            )

        agent.event_history.register_action(act)
        #agent.event_history.current_record.action = None
        #agent.event_history.cursor=0
        #agent.event_history.cycles.append(agent.event_history.cycles[-1])
        result = await agent.execute(next_command_name, next_command_args, assistant_reply_dict)
        #import pdb
        #pdb.set_trace()
        #result2 = await result
        print("result",result)

        
    except Exception as e:
        result = f"error {e}"
        raise e
    
    ret = f"RESULT:'{result}'"

    return ret
