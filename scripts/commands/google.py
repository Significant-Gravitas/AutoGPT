from operator import itemgetter
import json

from config import Config
from commands import Command
from duckduckgo_search import ddg
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

cfg = Config()

class Google(Command):
    def __init__(self):
        super().__init__()

    def _google_search(self, query, num_results=8):
        search_results = []
        
        for j in ddg(query, max_results=num_results):
            search_results.append(j)

        return json.dumps(search_results, ensure_ascii=False, indent=4)

    
    def _google_official_search(query, num_results=8):

        try:
            # Get the Google API key and Custom Search Engine ID from the config file
            api_key = cfg.google_api_key
            custom_search_engine_id = cfg.custom_search_engine_id

            # Initialize the Custom Search API service
            service = build("customsearch", "v1", developerKey=api_key)
            
            # Send the search query and retrieve the results
            result = service.cse().list(q=query, cx=custom_search_engine_id, num=num_results).execute()

            # Extract the search result items from the response
            search_results = result.get("items", [])
            
            # Create a list of only the URLs from the search results
            search_results_links = [item["link"] for item in search_results]

        except HttpError as e:
            # Handle errors in the API call
            error_details = json.loads(e.content.decode())
            
            # Check if the error is related to an invalid or missing API key
            if error_details.get("error", {}).get("code") == 403 and "invalid API key" in error_details.get("error", {}).get("message", ""):
                return "Error: The provided Google API key is invalid or missing."
            else:
                return f"Error: {e}"

        # Return the list of search result URLs
        return search_results_links

    def execute(self, **kwargs):
        query = itemgetter('input')(kwargs)

        # Check if the Google API key is set and use the official search method
        # If the API key is not set or has only whitespaces, use the unofficial search method
        if cfg.google_api_key and (cfg.google_api_key.strip() if cfg.google_api_key else None):
            return self._google_official_search(query)
        else:
            return self._google_search(query)
