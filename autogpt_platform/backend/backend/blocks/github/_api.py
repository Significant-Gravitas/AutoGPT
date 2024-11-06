from urllib.parse import urlparse

import requests

from ._auth import GithubCredentials


class GitHubAPI:
    def __init__(self, credentials: GithubCredentials):
        self.credentials = credentials

    @staticmethod
    def _validate_github_url(url: str) -> None:
        parsed_url = urlparse(url)
        if parsed_url.netloc != "github.com":
            raise ValueError("The input URL must be a valid GitHub URL.")

    @staticmethod
    def _convert_to_api_url(url: str) -> str:
        """
        Converts a standard GitHub URL to the corresponding GitHub API URL.
        Handles repository URLs, issue URLs, pull request URLs, and more.
        """
        GitHubAPI._validate_github_url(url)
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

    def _get_headers(self) -> dict:
        return {
            "Authorization": self.credentials.bearer(),
            "Accept": "application/vnd.github.v3+json",
        }

    def get(self, url: str, **kwargs) -> requests.Response:
        api_url = self._convert_to_api_url(url)
        headers = self._get_headers()
        response = requests.get(api_url, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    def post(self, url: str, json=None, **kwargs) -> requests.Response:
        api_url = self._convert_to_api_url(url)
        headers = self._get_headers()
        response = requests.post(api_url, headers=headers, json=json, **kwargs)
        response.raise_for_status()
        return response

    def delete(self, url: str, json=None, **kwargs) -> requests.Response:
        api_url = self._convert_to_api_url(url)
        headers = self._get_headers()
        response = requests.delete(api_url, headers=headers, json=json, **kwargs)
        response.raise_for_status()
        return response
