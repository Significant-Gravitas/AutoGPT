import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional

from browse import scrape_text, scrape_links, summarize_text, TextSummary
from config import Config
from duckduckgo_search import ddg
import logging

cfg = Config()


@dataclass
class SearchResult:
    query: str
    answer: str
    source: str


def google_search(query, num_results=8):
    search_results = []
    for j in ddg(query, max_results=num_results):
        search_results.append(j)

    return json.dumps(search_results, ensure_ascii=False, indent=4)


def google_official_search(query, num_results=8) -> Optional[List[str]]:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import json

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
        # Return the list of search result URLs
        return search_results_links
    except HttpError as e:
        # Handle errors in the API call
        error_details = json.loads(e.content.decode())

        # Check if the error is related to an invalid or missing API key
        if error_details.get("error", {}).get("code") == 403 and "invalid API key" in error_details.get("error",
                                                                                                        {}).get(
            "message", ""):
            logging.error("Error: The provided Google API key is invalid or missing.")
        else:
            logging.error(f"Error: {e}")
        return None


async def browse_website(url, question):
    summary = await get_text_summary(url, question)
    links = get_hyperlinks(url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]

    result = f"""Website Content Summary: {summary}\n\nLinks: {links}"""

    return result


async def get_text_summary(url, question) -> str:
    result = await get_text_summaries(url, question)
    if result:
        return """ "Result" : """ + result.answer
    else:
        return """ "Result" : """ + 'Failed'


async def get_text_summaries(urls: List[str] | str, question: str, num_sources: int = 1, max_tokens: int = 300) -> \
        Optional[
            List[SearchResult] | SearchResult]:
    results = []
    if isinstance(urls, str):
        urls = [urls]

    async def summarize_source(url: str):
        text = scrape_text(url)
        if not text:
            return None
        logging.debug(f'Summarizing {url}')
        return await summarize_text(text=text, question=question, source=url, max_tokens=max_tokens)

    results = []

    async def add_summaries(start: int, end: int):
        tasks = [
            summarize_source(url) for url in urls[start:end]]
        gathered_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend([
            SearchResult(source=r.source, query=question, answer=r.summary) for r in gathered_results if
            isinstance(r, TextSummary)
        ])

    start_source_index = 0
    for end_source_index in range(min(len(urls), num_sources), len(urls) + 1):
        await add_summaries(start_source_index, end_source_index)
        if len(results) >= num_sources:
            break
        start_source_index = end_source_index

    if not results:
        return None
    elif len(results) == 1:
        return results[0]
    else:
        return results


def get_hyperlinks(url):
    link_list = scrape_links(url)
    return link_list


async def get_search_results(query, return_results: int = 1, num_search: int = 8, max_tokens=300) -> Optional[
    List[SearchResult] | SearchResult]:
    links = google_official_search(query, num_search)
    return await get_text_summaries(links, question=query, num_sources=return_results, max_tokens=max_tokens)
