import json
import logging
import time
from typing import Iterator

from duckduckgo_search import DDGS
from pydantic import BaseModel

from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.config.config import Config
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import ConfigurationError, InvalidArgumentError

DUCKDUCKGO_MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    title: str
    url: str
    excerpt: str = ""


class WebSearchComponent(DirectiveProvider, CommandProvider):
    """Provides commands to search the web."""

    def __init__(self, config: Config):
        self.legacy_config = config

        if (
            not self.legacy_config.google_api_key
            or not self.legacy_config.google_custom_search_engine_id
        ):
            logger.info(
                "Configure google_api_key and custom_search_engine_id "
                "to use Google API search."
            )

    def get_resources(self) -> Iterator[str]:
        yield "Internet access for searches and information gathering."

    def get_commands(self) -> Iterator[Command]:
        yield self.web_search

        if (
            self.legacy_config.google_api_key
            and self.legacy_config.google_custom_search_engine_id
        ):
            yield self.google

    @command(
        ["web_search", "search"],
        "Searches the web",
        {
            "query": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The search query",
                required=True,
            ),
            "num_results": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="The number of results to return",
                minimum=1,
                maximum=10,
                required=False,
            ),
        },
    )
    def web_search(self, query: str, num_results: int = 8) -> list[SearchResult]:
        """Return the results of a Google search

        Args:
            query (str): The search query.
            num_results (int): The number of results to return.

        Returns:
            str: The results of the search.
        """
        if not query:
            raise InvalidArgumentError("'query' must be non-empty")

        search_results = []
        attempts = 0

        while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
            search_results = DDGS().text(query, max_results=num_results)

            if search_results:
                break

            time.sleep(1)
            attempts += 1

        return [
            SearchResult(
                title=r["title"],
                excerpt=r.get("body", ""),
                url=r["href"],
            )
            for r in search_results
        ]

    @command(
        ["google"],
        "Google Search",
        {
            "query": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The search query",
                required=True,
            ),
            "num_results": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="The number of results to return",
                minimum=1,
                maximum=10,
                required=False,
            ),
        },
    )
    def google(self, query: str, num_results: int = 8) -> list[SearchResult]:
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
            api_key = self.legacy_config.google_api_key
            custom_search_engine_id = self.legacy_config.google_custom_search_engine_id
            assert api_key and custom_search_engine_id  # checked in get_commands()

            # Initialize the Custom Search API service
            service = build("customsearch", "v1", developerKey=api_key)

            # Send the search query and retrieve the results
            result = (
                service.cse()
                .list(q=query, cx=custom_search_engine_id, num=num_results)
                .execute()
            )

            # Extract the search result items from the response
            return [
                SearchResult(
                    title=r.get("title", ""),
                    excerpt=r.get("snippet", ""),
                    url=r["link"],
                )
                for r in result.get("items", [])
                if "link" in r
            ]

        except HttpError as e:
            # Handle errors in the API call
            error_details = json.loads(e.content.decode())

            # Check if the error is related to an invalid or missing API key
            if error_details.get("error", {}).get("code") == 403 and (
                "invalid API key" in error_details["error"].get("message", "")
            ):
                raise ConfigurationError(
                    "The provided Google API key is invalid or missing."
                )
            raise
