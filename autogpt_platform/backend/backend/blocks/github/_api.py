from urllib.parse import urlparse

from backend.blocks.github._auth import (
    GithubCredentials,
    GithubFineGrainedAPICredentials,
)
from backend.util.request import Requests


def _convert_to_api_url(url: str) -> str:
    """
    Converts a standard GitHub URL to the corresponding GitHub API URL.
    Handles repository URLs, issue URLs, pull request URLs, and more.
    """
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip("/").split("/")

    if len(path_parts) >= 2:
        owner, repo = path_parts[0], path_parts[1]
        api_base = f"https://api.github.com/repos/{owner}/{repo}"

        if len(path_parts) > 2:
            additional_path = "/".join(path_parts[2:])
            api_url = f"{api_base}/{additional_path}"
        else:
            # Repository base URL
            api_url = api_base
    else:
        raise ValueError("Invalid GitHub URL format.")

    return api_url


def _get_headers(credentials: GithubCredentials) -> dict[str, str]:
    return {
        "Authorization": credentials.auth_header(),
        "Accept": "application/vnd.github.v3+json",
    }


def get_api(
    credentials: GithubCredentials | GithubFineGrainedAPICredentials,
    convert_urls: bool = True,
) -> Requests:
    return Requests(
        trusted_origins=["https://api.github.com", "https://github.com"],
        extra_url_validator=_convert_to_api_url if convert_urls else None,
        extra_headers=_get_headers(credentials),
    )
