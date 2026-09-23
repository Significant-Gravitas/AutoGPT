from typing import Callable, Optional, overload
from urllib.parse import urlparse

from backend.blocks.github._auth import (
    GithubCredentials,
    GithubFineGrainedAPICredentials,
)
from backend.util.request import URL, Requests

GITHUB_API_URL = "https://api.github.com"


@overload
def _convert_to_api_url(url: str) -> str: ...


@overload
def _convert_to_api_url(url: URL) -> URL: ...


def _convert_to_api_url(url: str | URL) -> str | URL:
    """
    Converts a standard GitHub URL to the corresponding GitHub API URL.
    Handles repository URLs, issue URLs, pull request URLs, and more.
    """
    if url_as_str := isinstance(url, str):
        url = urlparse(url)

    path_parts = url.path.strip("/").split("/")

    if len(path_parts) >= 2:
        owner, repo = path_parts[0], path_parts[1]
        api_base = f"{GITHUB_API_URL}/repos/{owner}/{repo}"

        if len(path_parts) > 2:
            additional_path = "/".join(path_parts[2:])
            api_url = f"{api_base}/{additional_path}"
        else:
            # Repository base URL
            api_url = api_base
    else:
        raise ValueError("Invalid GitHub URL format.")

    return api_url if url_as_str else urlparse(api_url)


def _get_headers(credentials: GithubCredentials) -> dict[str, str]:
    return {
        "Authorization": credentials.auth_header(),
        "Accept": "application/vnd.github.v3+json",
    }


def convert_comment_url_to_api_endpoint(comment_url: str) -> str:
    """
    Converts a GitHub comment URL (web interface) to the appropriate API endpoint URL.

    Handles:
    1. Issue/PR comments: #issuecomment-{id}
    2. PR review comments: #discussion_r{id}

    Returns the appropriate API endpoint path for the comment.
    """
    # First, check if this is already an API URL
    parsed_url = urlparse(comment_url)
    if parsed_url.hostname == "api.github.com":
        return comment_url

    # Replace pull with issues for comment endpoints
    if "/pull/" in comment_url:
        comment_url = comment_url.replace("/pull/", "/issues/")

    # Handle issue/PR comments (#issuecomment-xxx)
    if "#issuecomment-" in comment_url:
        base_url, comment_part = comment_url.split("#issuecomment-")
        comment_id = comment_part

        # Extract repo information from base URL
        parsed_url = urlparse(base_url)
        path_parts = parsed_url.path.strip("/").split("/")
        owner, repo = path_parts[0], path_parts[1]

        # Construct API URL for issue comments
        return f"{GITHUB_API_URL}/repos/{owner}/{repo}/issues/comments/{comment_id}"

    # Handle PR review comments (#discussion_r)
    elif "#discussion_r" in comment_url:
        base_url, comment_part = comment_url.split("#discussion_r")
        comment_id = comment_part

        # Extract repo information from base URL
        parsed_url = urlparse(base_url)
        path_parts = parsed_url.path.strip("/").split("/")
        owner, repo = path_parts[0], path_parts[1]

        # Construct API URL for PR review comments
        return f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls/comments/{comment_id}"

    # If no specific comment identifiers are found, use the general URL conversion
    return _convert_to_api_url(comment_url)


def get_api(
    credentials: GithubCredentials | GithubFineGrainedAPICredentials,
    convert_urls: bool = True,
) -> Requests:
    return Requests(
        trusted_origins=[GITHUB_API_URL, "https://github.com"],
        extra_url_validator=_convert_to_api_url if convert_urls else None,
        extra_headers=_get_headers(credentials),
    )


async def get_paginated(
    api: Requests,
    url: str,
    *,
    limit: int,
    params: Optional[dict[str, str]] = None,
    keep: Optional[Callable[[dict], bool]] = None,
    max_page_size: int = 100,
    start_page: int = 1,
) -> list[dict]:
    """
    Fetches items from a paginated GitHub list endpoint until `limit` items
    are collected or no pages are left. Items are filtered by `keep` (if given)
    before counting towards `limit`. Fetching starts at `start_page`.
    """
    items: list[dict] = []
    page_size = max_page_size if keep else min(limit, max_page_size)
    page = start_page
    while len(items) < limit:
        response = await api.get(
            url,
            params={**(params or {}), "per_page": str(page_size), "page": str(page)},
        )
        batch = response.json()
        items.extend(item for item in batch if keep is None or keep(item))
        if len(batch) < page_size:
            break
        page += 1
    return items[:limit]
