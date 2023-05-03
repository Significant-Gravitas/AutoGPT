import requests
import json
import os
import re


def clean_text(text: str) -> str:
    cleaned_text = re.sub("<[^>]*>", "", text)  # Remove HTML tags
    cleaned_text = cleaned_text.replace(
        "\\n", " "
    )  # Replace newline characters with spaces
    return cleaned_text


def _bing_search(query: str, num_results=8) -> str:
    """
    Perform a Bing search and return the results as a JSON string.
    """
    subscription_key = os.getenv("BING_API_KEY")

    # Bing Search API endpoint
    search_url = "https://api.bing.microsoft.com/v7.0/search"

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {
        "q": query,
        "count": num_results,
        "textDecorations": True,
        "textFormat": "HTML",
    }
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    # Extract the search result items from the response
    web_pages = search_results.get("webPages", {})
    search_results = web_pages.get("value", [])

    # Create a list of search result dictionaries with 'title', 'href', and 'body' keys
    search_results_list = [
        {
            "title": clean_text(item["name"]),
            "href": item["url"],
            "body": clean_text(item["snippet"]),
        }
        for item in search_results
    ]

    # Return the search results as a JSON string
    return json.dumps(search_results_list, ensure_ascii=False, indent=4)
