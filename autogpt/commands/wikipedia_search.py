"""Wikipedia search command for Autogpt."""
from __future__ import annotations

import json
import re
from urllib.parse import quote

import requests

from autogpt.config import Config
from autogpt.logs import logger

HTML_TAG_CLEANER = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


def wikipedia_search(query: str, num_results: int = 5) -> str:
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
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&utf8=1&formatversion=2&srsearch={quote(query)}"
    with requests.Session() as session:
        session.headers.update({"User-Agent": Config().user_agent})
        session.headers.update({'Accept': 'application/json'})
        results = session.get(search_url)
        try:
            results = results.json()
            r = []
            for item in results['query']['search']:
                summary = re.sub(HTML_TAG_CLEANER, '', item["snippet"])
                r.append({
                    "title": item["title"],
                    "summary": summary,
                    "url": f"http://en.wikipedia.org/?curid={item['pageid']}",
                })
                if len(r) == num_results:
                    break
        except Exception as e:
            logger.debug(
                f"'wikipedia_search' on query: {query} raised exception: {e}")
            return f"Error: {e}"

    return json.dumps(r, ensure_ascii=False, indent=4)
