"""
Searching with googleapi
"""
from typing import List

import os
import googleapiclient.discovery

from .registry import ability

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
    service = googleapiclient.discovery.build(
        "customsearch",
        "v1",
        developerKey=os.getenv("GOOGLE_API_KEY"))
    
    response_snippets = []
    response = service.cse().list(
        q=query,
        cx=os.getenv("GOOGLE_CSE_ID")
    ).execute()

    for result in response["items"]:
        response_snippets.append({
            "url": result["formattedUrl"],
            "snippet": result["snippet"]
        })

    return response_snippets