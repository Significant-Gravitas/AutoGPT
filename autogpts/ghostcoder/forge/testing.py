import asyncio
import os
from pathlib import Path

import tiktoken

from forge.agent import ForgeAgent
from forge.db import ForgeDatabase
from forge.sdk import LocalWorkspace
import forge.sdk.db
from forge.sdk.abilities.coding.external_coder import fix_code
from forge.sdk.abilities.coding.split_coder import write_code

workspace = LocalWorkspace("/home/albert/repos/albert/AutoGPT/autogpts/ghostcoder/agbenchmark_config/workspace")
database = ForgeDatabase("sqlite:///agent.db", debug_enabled=False)
agent = ForgeAgent(database=database, workspace=workspace)

result = asyncio.get_event_loop().run_until_complete(write_code(agent, "23fb544e-82a1-4578-9fa8-101274ba8733", "ef50d501-332a-4a77-80a8-971971bf881a", "battleship.py"))
