"""
Searching with googleapi
"""
from typing import List

import os
import json
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
    output_type="str",
)
async def google_search(agent, task_id: str, query: str) -> str:
    """
    Return list of snippets from google search
    """

    resp_message = "No results found"

    try:
        service = googleapiclient.discovery.build(
            "customsearch",
            "v1",
            developerKey=os.getenv("GOOGLE_API_KEY"))
        
        
        response = service.cse().list(
            q=query,
            cx=os.getenv("GOOGLE_CSE_ID")
        ).execute()

        results = response["items"]

        # adding safe message code from latest forge sdk
        try:
            mem_message = json.dumps(
                [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
            )
        except Exception as err:
            logger.error("error making safe_message json, using dict and str instead: {err}")

            resp_list = []
            for result in response["items"]:
                resp_list.append({
                    "url": result["formattedUrl"],
                    "snippet": result["snippet"]
                })

            mem_message = str(resp_list)

        add_ability_memory(task_id, mem_message, "google_search")

        resp_message = f"{len(response['items'])} Results from query '{query}' stored in memory"
    except Exception as err:
        logger.error(f"google_search failed: {err}")
        raise err

    return resp_message