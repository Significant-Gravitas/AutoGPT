from typing import Dict
from backend.data.block import Block
from backend.util.request import requests


class Slant3DBlockBase(Block):
    """Base block class for Slant3D API interactions"""

    BASE_URL = "https://www.slant3dapi.com/api"

    def _get_headers(self, api_key: str) -> Dict[str, str]:
        return {"api-key": api_key, "Content-Type": "application/json"}

    def _make_request(self, method: str, endpoint: str, api_key: str, **kwargs) -> Dict:
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.request(
            method=method, url=url, headers=self._get_headers(api_key), **kwargs
        )

        if not response.ok:
            error_msg = response.json().get("error", "Unknown error")
            raise RuntimeError(f"API request failed: {error_msg}")

        return response.json()
