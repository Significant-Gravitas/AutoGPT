"""
Memstore methods used by tools
"""
import os
from forge.sdk.memory.memstore import ChromaMemStore
from ..forge_log import ForgeLogger

logger = ForgeLogger(__name__)

async def add_ability_memory(task_id: str, document: str, ability_name: str) -> None:
    logger.info(f"🧠 Adding ability '{ability_name}' memory for task {task_id}")
    try:
        chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}/chromadb/"
        memory = ChromaMemStore(chromadb_path)
        await memory.add(
            task_id=task_id,
            document=document,
            metadatas={"function": ability_name}
        )
    except Exception as err:
        logger.error(f"add_memory failed: {err}")