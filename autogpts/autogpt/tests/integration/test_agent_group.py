import asyncio
import logging
import uuid
import requests
import pytest
from unittest.mock import patch
from autogpt.agents.agent_group import AgentGroup, create_agent_member
from autogpt.core.resource.model_providers.openai import OPEN_AI_CHAT_MODELS, _tool_calls_compat_extract_calls
from autogpt.core.resource.model_providers.schema import ChatModelResponse
from forge.sdk.model import TaskRequestBody 
from autogpt.core.resource.model_providers.schema import AssistantChatMessage
logging.basicConfig(
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def run_tasks(agents):
    for agent in agents:
        asyncio.create_task(agent.run_tasks())

async def main():
    ceo = await create_agent_member(
        role="ceo",
        initial_prompt="you are ceo of a software game company"
    )

    hr_lead = await create_agent_member(
        role="hr_lead",
        initial_prompt="you are hr_lead of a company You'll recruite agents when we need it",
        boss=ceo,
        create_agent=True
    )

    ceo.recruiter = hr_lead

    agentGroup = AgentGroup(
        leader=ceo
    )

    await agentGroup.create_task(TaskRequestBody(input="create best shooter game in the world", additional_input={}))

    await run_tasks([ceo, hr_lead])

    while True:
        await asyncio.sleep(1)  
        running_tasks = [task for task in asyncio.all_tasks() if not task.done()]
        if not running_tasks:
            break  
def empty_file(file_path):
    try:
        with open(file_path, 'w') as f:
            f.truncate(0)
        print(f"The file '{file_path}' has been emptied successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

@pytest.mark.asyncio
@pytest.mark.vcr("/home/mahdi/all/repositories/github.com/autogpt/AutoGPT/autogpts/autogpt/tests/vcr_cassettes/test_agent_group_happy_scenario/test_agent_group_happy_scenario.yaml")
async def test_agent_group_happy_scenario():
    empty_file("/home/mahdi/all/repositories/github.com/autogpt/AutoGPT/autogpts/autogpt/agetn_group.db")
    expected_uuids_for_tasks = ['11111111-1111-1111-1111-111111111121', '11111111-1111-1111-1111-111111111122','11111111-1111-1111-1111-111111111123','11111111-1111-1111-1111-111111111124']
    expected_uuids_for_agents = ['33333333-3333-3333-3333-333333333331', '33333333-3333-3333-3333-333333333332']
    with patch('forge.sdk.db.uuid4', side_effect=expected_uuids_for_tasks):
        with patch('autogpt.agents.agent_member.uuid4', side_effect=expected_uuids_for_agents):
            await main()
