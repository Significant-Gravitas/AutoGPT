from __future__ import annotations

import json
import time

from duckduckgo_search import DDGS

from forge.sdk.agent import Agent

from ..registry import ActionParameter, action

DUCKDUCKGO_MAX_ATTEMPTS = 3


@action(
    name="web_search",
    description="Searches the web",
    parameters=[
        ActionParameter(
            name="query",
            description="The search query",
            type="string",
            required=True,
        )
    ],
    output_type="list[str]",
)
async def web_search(agent: Agent, task_id: str, query: str) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    attempts = 0
    num_results = 8

    while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
        if not query:
            return json.dumps(search_results)

        search_results = DDGS().text(query, max_results=num_results)

        if search_results:
            break

        time.sleep(1)
        attempts += 1

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)


def safe_google_results(results: str | list) -> str:
    """
        Return the results of a Google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps(
            [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message
