"""
Memory tool for document search
"""
from typing import List

import os
from .registry import ability

from forge.sdk.memory.memstore import ChromaMemStore

@ability(
    name="mem_search",
    description="Search documents stored in memory",
    parameters=[
        {
            "name": "query",
            "description": "search query",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[str]",
)
async def mem_search(agent, task_id: str, query: str) -> List[str]:
    chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}"

    memory = ChromaMemStore(chromadb_path)
    memory_resp = memory.query(
        task_id=task_id,
        query=query
    )

    return memory_resp["documents"][0]


