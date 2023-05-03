"""Wikipedia search command for Autogpt."""
from __future__ import annotations

import json
import re
from urllib.parse import quote

import requests

HTML_TAG_CLEANER = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


def _wikipedia_search(query: str, num_results: int = 5) -> str | list[str]:
    """Return the results of a Wikipedia search
    Args:
        query (str): The search query.
        num_results (int): The number of results to return.
    Returns:
        str: The results of the search. The resulting string is a `json.dumps`
             of a list of len `num_results` containing dictionaries with the
             following structure: `{'title': <title>, 'summary': <summary>,
             'url': <url to relevant page>}`
    """
    search_url = (
        "https://en.wikipedia.org/w/api.php?action=query&"
        "format=json&list=search&utf8=1&formatversion=2&"
        f"srsearch={quote(query)}"
    )
    with requests.Session() as session:
        session.headers.update({"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; "
                                               "Win64; x64) AppleWebKit/537.36 "
                                               "(KHTML, like Gecko) Chrome/"
                                               "112.0.5615.49 Safari/537.36")})
        session.headers.update({"Accept": "application/json"})
        results = session.get(search_url)
        items = []
        try:
            results = results.json()
            for item in results["query"]["search"]:
                summary = re.sub(HTML_TAG_CLEANER, "", item["snippet"])
                items.append(
                    {
                        "title": item["title"],
                        "summary": summary,
                        "url": f"http://en.wikipedia.org/?curid={item['pageid']}",
                    }
                )
                if len(items) == num_results:
                    break
        except Exception as e:
            return f"'wikipedia_search' on query: {query} raised exception: {e}"

    return json.dumps(items, ensure_ascii=False, indent=4)