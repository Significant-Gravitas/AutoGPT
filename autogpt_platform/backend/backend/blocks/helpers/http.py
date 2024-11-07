from typing import Any, Optional

import requests


class GetRequest:
    @classmethod
    def get_request(
        cls, url: str, headers: Optional[dict] = None, json: bool = False
    ) -> Any:
        if headers is None:
            headers = {}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json() if json else response.text
