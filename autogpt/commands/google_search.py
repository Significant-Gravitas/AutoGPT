"""Google search command for Autogpt."""
from __future__ import annotations

import json

import requests
from duckduckgo_search import ddg

from autogpt.commands.command import command
from autogpt.config import Config

CFG = Config()


@command(
    "google",
    "Google Search",
    '"query": "<query>"',
    not (CFG.google_api_key or CFG.serper_api_key),
)
def google_search(query: str, num_results: int = 8) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    if not query:
        return json.dumps(search_results)

    results = ddg(query, max_results=num_results)
    if not results:
        return json.dumps(search_results)

    for j in results:
        search_results.append(j)

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)


@command(
    "google",
    "Google Search",
    '"query": "<query>"',
    bool(CFG.serper_api_key),
    "Configure serper_api_key.",
)
def google_serper_search(query: str, num_results: int = 8) -> str:
    """Return the results of a Google search using the Serper Google Search API.
    Use this if you're running into rate limits or want to use the full search results from google.com, so you can
    leverage the Google knowledge graph, featured snippets and 'related searches' information.

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """

    if not query:
        return json.dumps({})

    try:
        api_key = CFG.serper_api_key
        headers = {
            "X-API-KEY": api_key or "",
            "Content-Type": "application/json",
        }
        params = {"q": query, "num": num_results}
        response = requests.post(
            "https://google.serper.dev/search", headers=headers, params=params
        )
        response.raise_for_status()
        search_results = response.json()
        results = json.dumps(search_results, ensure_ascii=False, indent=4)
        return safe_google_results(results)

    except requests.exceptions.HTTPError as e:
        if e.response.text:
            return f"Serper API Error: {e.response.text}"
        return f"HTTP Error: {e}"

    except Exception as e:
        return f"Error: {e}"


@command(
    "google",
    "Google Search",
    '"query": "<query>"',
    bool(CFG.google_api_key),
    "Configure google_api_key.",
)
def google_official_search(query: str, num_results: int = 8) -> str | list[str]:
    """Return the results of a Google search using the official Google API

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """

    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    try:
        # Get the Google API key and Custom Search Engine ID from the config file
        api_key = CFG.google_api_key
        custom_search_engine_id = CFG.custom_search_engine_id

        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=api_key)

        # Send the search query and retrieve the results
        result = (
            service.cse()
            .list(q=query, cx=custom_search_engine_id, num=num_results)
            .execute()
        )

        # Extract the search result items from the response
        search_results = result.get("items", [])

        # Create a list of only the URLs from the search results
        search_results_links = [item["link"] for item in search_results]

    except HttpError as e:
        # Handle errors in the API call
        error_details = json.loads(e.content.decode())

        # Check if the error is related to an invalid or missing API key
        if error_details.get("error", {}).get(
            "code"
        ) == 403 and "invalid API key" in error_details.get("error", {}).get(
            "message", ""
        ):
            return "Error: The provided Google API key is invalid or missing."
        else:
            return f"Error: {e}"
    # google_result can be a list or a string depending on the search results

    # Return the list of search result URLs
    return safe_google_results(search_results_links)


def safe_google_results(results: str | list) -> str:
    """
        Return the results of a google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps(
            [result.enocde("utf-8", "ignore") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message
