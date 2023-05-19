"""Google search command for Autogpt."""
from __future__ import annotations

import json

from duckduckgo_search import ddg

from autogpt.commands.command import command
from autogpt.config import Config
#import os
import requests

CFG = Config()
from . import URL_MEMORY

@command("google", "Google Search", '"query": "<query>"', not CFG.google_api_key)
def google_search(query: str, num_results: int = 10) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    global URL_MEMORY
    search_results = []
    if not query:
        return json.dumps(search_results)

    results = ddg(query, max_results=num_results)
    if not results:        
        return json.dumps(search_results)

    for j in results:
        url_alias = f'URL_{len(URL_MEMORY)}'
        URL_MEMORY[url_alias] = j['href']
        j['href'] = url_alias
        del j['body']
        search_results.append(j)

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)


@command(
    "google",
    "Google Search",
    '"query": "<query>"',
    bool(CFG.google_api_key) and bool(CFG.custom_search_engine_id),
    "Configure google_api_key and custom_search_engine_id.",
)
def google_official_search(query: str, num_results: int = 10) -> str | list[str]:
    """Return the results of a Google search using the official Google API

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """

    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    global URL_MEMORY
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
        search_results = []
        # Extract the search result items from the response
        results = result.get("items", [])

        # Create a list of only the URLs from the search results
        #search_results_links = [item["link"] for item in search_results]
        
        for res in results:
            url_alias = f'URL_{len(URL_MEMORY)}'
            URL_MEMORY[url_alias] = res['link']
            res['link'] = url_alias
            res = {k:v for k,v in res.items() if k in ['title', 'link', 'snippet']}            
            search_results.append(res)

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
    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)


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
            [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message

@command(
    "google_search_place",
    "Google Search place",
    '"place_name": "<place_name>"',
    bool(CFG.google_api_key),
)
def google_search_place(place_name: str, num_results: int = 10) -> str:
    api_key = CFG.google_api_key
    url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={place_name}&inputtype=textquery&fields=place_id,name&key={api_key}"
    result = requests.get(url)
    json_obj = result.json()

    candidates = []
    for candidate in json_obj['candidates'][:num_results]:
        place_id = candidate['place_id']  # Get the place ID of the first result
        place_details = get_place_details(place_id)
        place_details['address_components'] = [comp['short_name'] for comp in place_details['address_components'][1:-2]]
        place_details['location'] = place_details['geometry']['location']
        candidate.update(place_details)
        del candidate['place_id'], candidate['geometry']
        candidates.append(candidate)
    return json.dumps(candidates, ensure_ascii=False)

def get_place_details(place_id):
    api_key = CFG.google_api_key
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=geometry/location,rating,types,address_components&key={api_key}"
    result = requests.get(url)
    json_obj = result.json()
    return json_obj['result']

@command(
    "google_search_nearby_places",
    "Google Search nearby places. You must first obtain 'latitude' and 'longitude' before calling this command.",
    '"latitude": "<latitude>", "longitude": "<longitude>", "radius":"1000"',
    bool(CFG.google_api_key),
)
def google_search_nearby_places(latitude, longitude, radius: int=1000, num_results: int = 10) -> str:
    api_key = CFG.google_api_key
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius={radius}&key={api_key}"
    result = requests.get(url)
    json_obj = result.json()
    
    candidates = []
    for candidate in json_obj['results'][:num_results]:
        place_id = candidate['place_id']  # Get the place ID of the first result
        place_details = get_place_details(place_id)
        candidate = {k:v for k, v in candidate.items() if k in ['name', 'place_id']}
        place_details['address_components'] = [comp['short_name'] for comp in place_details['address_components'][1:-2]]
        place_details['location'] = place_details['geometry']['location']
        candidate.update(place_details)
        del candidate['place_id'], candidate['geometry']
        candidates.append(candidate)
    return json.dumps(candidates, ensure_ascii=False)
