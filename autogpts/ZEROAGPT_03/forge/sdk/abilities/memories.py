"""
Memory tool for document search
"""
from typing import List

import os
from ..forge_log import ForgeLogger
from .registry import ability

from forge.sdk.memory.memstore import ChromaMemStore

from ..ai_memory import AIMemory

logger = ForgeLogger(__name__)

@ability(
    name="add_file_memory",
    description="Add content of file to your memory",
    parameters=[
        {
            "name": "file_name",
            "description": "File name of file to add",
            "type": "string",
            "required": True,
        }
    ],
    output_type="None",
)
async def add_file_memory(agent, task_id: str, file_name: str) -> None:
    logger.info(f"🧠 Adding {file_name} to memory for task {task_id}")
    try:
        cwd = agent.workspace.get_cwd_path(task_id)
        chroma_dir = f"{cwd}/chromadb/"

        open_file = agent.workspace.read(task_id=task_id, path=file_name)
        open_file_str = open_file.decode()

        memory = ChromaMemStore(chroma_dir)
        memory.add(
            task_id=task_id,
            document=open_file_str,
            metadatas={"filename": file_name}
        )
    except Exception as err:
        logger.error(f"add_file_memory failed: {err}")

@ability(
    name="read_file_from_memory",
    description="Read file stored in your memory",
    parameters=[
        {
            "name": "file_name",
            "description": "File name of file to add",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def read_file_from_memory(agent, task_id: str, file_name: str) -> str:
    mem_doc = f"{file_name} not found in database"

    try:
        # find doc in chromadb
        cwd = agent.workspace.get_cwd_path(task_id)
        chroma_dir = f"{cwd}/chromadb/"

        memory = ChromaMemStore(chroma_dir)
        memory_resp = memory.query(
            task_id=task_id,
            query=file_name
        )

        # get the most relevant document and shrink to 255
        if len(memory_resp["documents"][0]) > 0:
            mem_doc = memory_resp["documents"][0][0][:255]
    except Exception as err:
        logger.error(f"mem_search failed: {err}")
        raise err
    
    return mem_doc

@ability(
    name="mem_search",
    description="query your memory for relevant stored documents",
    parameters=[
        {
            "name": "query",
            "description": "search query",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def mem_search(agent, task_id: str, query: str) -> str:
    mem_doc = "No documents found"

    try:
        # find doc in chromadb
        cwd = agent.workspace.get_cwd_path(task_id)
        chroma_dir = f"{cwd}/chromadb/"

        memory = ChromaMemStore(chroma_dir)
        memory_resp = memory.query(
            task_id=task_id,
            query=query
        )

        # get the most relevant document and shrink to 255
        if len(memory_resp["documents"][0]) > 0:
            mem_doc = memory_resp["documents"][0][0][:255]
    except Exception as err:
        logger.error(f"mem_search failed: {err}")
        raise err
    
    return mem_doc

@ability(
    name="mem_qna",
    description="Query for a relevant document in your memory question it",
    parameters=[
        {
            "name": "doc_query",
            "description": "query or keyword to find document in database",
            "type": "string",
            "required": True,
        },
        {
            "name": "doc_question",
            "description": "question about document",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def mem_qna(agent, task_id: str, doc_query: str, doc_question: str):
    mem_doc = "No documents found"
    try:
        aimem = AIMemory(
            agent.workspace,
            task_id,
            doc_query,
            doc_question,
            "gpt-3.5-turbo-16k"
        )

        aimem.get_doc()
        
        mem_doc = await aimem.query_doc_ai()
    except Exception as err:
        logger.error(f"mem_qna failed: {err}")
        raise err
    
    return mem_doc