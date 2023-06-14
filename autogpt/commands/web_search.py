"""Google search command for Autogpt."""
from __future__ import annotations

import json
import time
from itertools import islice

from duckduckgo_search import DDGS

from autogpt.agent.agent import Agent
from autogpt.command_decorator import command

DUCKDUCKGO_MAX_ATTEMPTS = 3


@command(
    "web_search",
    "Web Search",
    arguments={"query": {"type": "string", "description": "The search query"}},
)
def web_search(query: str, agent: Agent, num_results: int = 8) -> str:
    """Return the results of a Web search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    attempts = 0

    while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
        if not query:
            return json.dumps(search_results)

        results = DDGS().text(query)
        search_results = list(islice(results, num_results))

        if search_results:
            break

        time.sleep(1)
        attempts += 1

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_web_search_results(results)


def safe_web_search_results(results: str | list) -> str:
    """
        Return the results of a web_search search in a safe format.

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
