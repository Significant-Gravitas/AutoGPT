from typing import Any, Optional

from backend.util.request import Requests


class GetRequest:
    @classmethod
    async def get_request(
        cls, url: str, headers: Optional[dict] = None, json: bool = False
    ) -> Any:
        if headers is None:
            headers = {}
        response = await Requests().get(url, headers=headers)
        if json:
            return response.json()
        else:
            return response.text()
