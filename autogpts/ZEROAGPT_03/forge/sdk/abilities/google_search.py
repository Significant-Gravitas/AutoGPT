"""
Searching with googleapi
"""
from typing import List

import os
import googleapiclient.discovery

from forge.sdk.memory.memstore_tools import add_ability_memory

from ..forge_log import ForgeLogger
from .registry import ability

logger = ForgeLogger(__name__)

@ability(
    name="google_search",
    description="Search the internet using Google",
    parameters=[
        {
            "name": "query",
            "description": "detailed search query",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[str]",
)
async def google_search(agent, task_id: str, query: str) -> List[str]:
    """
    Return list of snippets from google search
    """

    response_snippets = []

    try:
        service = googleapiclient.discovery.build(
            "customsearch",
            "v1",
            developerKey=os.getenv("GOOGLE_API_KEY"))
        
        
        response = service.cse().list(
            q=query,
            cx=os.getenv("GOOGLE_CSE_ID")
        ).execute()

        for result in response["items"]:
            response_snippets.append({
                "url": result["formattedUrl"],
                "snippet": result["snippet"]
            })

        add_ability_memory(task_id, str(response_snippets), "google_search")
    except Exception as err:
        logger.error(f"google_search failed: {err}")
        raise err

    return response_snippets