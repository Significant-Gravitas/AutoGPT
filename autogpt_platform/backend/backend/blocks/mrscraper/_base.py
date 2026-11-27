from typing import Any

from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials, Block, MediaFileType

from ._api import MrScraperRequest, send, store_screenshot


class MrScraperBlockBase(Block):
    """The upstream call and screenshot storage, as methods block tests can mock."""

    async def _send(
        self, credentials: APIKeyCredentials, request: MrScraperRequest
    ) -> Any:
        return await send(credentials, request)

    async def _store_screenshot(
        self, screenshot: str, execution_context: ExecutionContext
    ) -> MediaFileType:
        return await store_screenshot(screenshot, execution_context)
