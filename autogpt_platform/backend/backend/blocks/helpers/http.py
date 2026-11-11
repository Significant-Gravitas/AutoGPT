from typing import Any

from backend.util.request import Requests


class GetRequest:
    @classmethod
    async def get_request(
        cls, url: str, headers: dict | None = None, json: bool = False
    ) -> Any:
        if headers is None:
            headers = {}
        response = await Requests().get(url, headers=headers)
        if json:
            return response.json()
        else:
            return response.text()
