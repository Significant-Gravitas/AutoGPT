"""
HTTP access to the Exa Websets API (``https://api.exa.ai/websets/v0``).

The installed ``exa_py`` SDK validates responses against enums that are older
than Exa's live API (no ``pending`` webset or search status, no ``canceled``
import status, no ``failed`` monitor run status) and its request models lack
newer fields (webset ``title``/``exclude``, search ``recall``, update
``title``, preview ``search`` wrapper). Blocks that read or write those shapes
go through this module instead, so one malformed-by-the-SDK value can't crash
a run and every field the spec defines reaches Exa.
"""

from asyncio import sleep
from time import monotonic
from typing import Any

from backend.sdk import APIKeyCredentials, Block

from ._exa_rest import WEBSETS_API_BASE, QueryParams, add_param, exa_request
from ._exa_rest import path_segment as segment
from .helpers import merge_exa_cost
from .websets_types import ApiWebset


class ExaWebsetsBlockBase(Block):
    """Base for Websets blocks: one mockable entry point to the Websets API.

    ``_api`` records any ``costDollars`` the response carries, so billing
    works without every block remembering to do it. The ``Base`` suffix keeps
    the block loader from registering this class as a block.
    """

    async def _api(
        self,
        credentials: APIKeyCredentials,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = await exa_request(
            credentials,
            method,
            f"{WEBSETS_API_BASE}{path}",
            body=json,
            params=query_params(params),
        )
        merge_exa_cost(self, data)
        return data

    async def _get_webset(
        self, credentials: APIKeyCredentials, webset_id: str
    ) -> ApiWebset:
        data = await self._api(credentials, "GET", f"/websets/{segment(webset_id)}")
        return ApiWebset.model_validate(data)

    async def _wait_until_idle(
        self,
        credentials: APIKeyCredentials,
        webset_id: str,
        timeout: float,
        poll_interval: float = 5,
    ) -> ApiWebset:
        """Poll a webset until it is idle; raise ``TimeoutError`` past ``timeout``."""
        deadline = monotonic() + timeout
        while True:
            webset = await self._get_webset(credentials, webset_id)
            if webset.status == "idle":
                return webset
            if monotonic() >= deadline:
                raise TimeoutError(
                    f"Webset {webset_id} was still '{webset.status}' after "
                    f"{timeout:g} seconds."
                )
            await sleep(poll_interval)


def query_params(params: dict[str, Any] | None) -> QueryParams:
    """Turn ``{name: value}`` into Exa query params, dropping unset values."""
    query: QueryParams = []
    for name, value in (params or {}).items():
        add_param(query, name, value)
    return query
