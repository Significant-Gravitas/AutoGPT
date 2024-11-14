from typing import Any, Optional

from backend.util.request import requests


class GetRequest:
    @classmethod
    def get_request(
        cls, url: str, headers: Optional[dict] = None, json: bool = False
    ) -> Any:
        if headers is None:
            headers = {}
        response = requests.get(url, headers=headers)
        return response.json() if json else response.text
