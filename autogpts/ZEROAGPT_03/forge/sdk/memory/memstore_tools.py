"""
Memstore methods used by tools
"""
import os
from forge.sdk.memory.memstore import ChromaMemStore
from ..forge_log import ForgeLogger

logger = ForgeLogger(__name__)

def add_ability_memory(task_id: str, document: str, ability_name: str) -> None:
    """
    Add ability output to memory
    """
    logger.info(f"🧠 Adding ability '{ability_name}' memory for task {task_id}")
    try:
        chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}/chromadb/"
        memory = ChromaMemStore(chromadb_path)
        memory.add(
            task_id=task_id,
            document=document,
            metadatas={
                "function": ability_name,
                "task_id": task_id
            }
        )
    except Exception as err:
        logger.error(f"add_ability_memory failed: {err}")

def add_chat_memory(task_id: str, chat_msg: dict) -> None:
    """
    Add chat entry to memory
    """
    logger.info(f"🧠 Adding chat memory for task {task_id}")
    try:
        chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}/chromadb/"
        memory = ChromaMemStore(chromadb_path)
        memory.add(
            task_id=task_id,
            document=chat_msg["content"],
            metadatas={
                "role": chat_msg["role"],
                "task_id": task_id
            }
        )
    except Exception as err:
        logger.error(f"add_chat_memory failed: {err}")

def add_website_memory(task_id: str, url: str, content: str) -> None:
    """
    Add website to memory
    """
    logger.info(f"🧠 Adding website memory for task {task_id}")
    try:
        chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}/chromadb/"
        memory = ChromaMemStore(chromadb_path)
        memory.add(
            task_id=task_id,
            document=content,
            metadatas={
                "website": url,
                "task_id": task_id
            }
        )
    except Exception as err:
        logger.error(f"add_chat_memory failed: {err}")
