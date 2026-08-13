"""Thin async client for the Telegram Bot API.

Plain HTTPS JSON calls via httpx — the Bot API surface we need is small
enough that a dedicated SDK dependency isn't warranted.
"""

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"
_REQUEST_TIMEOUT_SECONDS = 30.0


class TelegramAPIError(Exception):
    """A Bot API call returned ok=false."""


class TelegramClient:
    def __init__(self, token: str):
        self._token = token
        # One pooled client for the adapter's lifetime (mirrors how Slack's
        # AsyncWebClient holds its session) — avoids a TCP+TLS handshake per
        # Bot API call.
        self._http = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS)

    async def call(self, method: str, **params: Any) -> Any:
        """POST a Bot API method; return its ``result`` or raise."""
        url = f"{_API_BASE}/bot{self._token}/{method}"
        resp = await self._http.post(url, json=_drop_none(params))
        payload = resp.json()
        if not payload.get("ok"):
            raise TelegramAPIError(
                f"{method} failed: {payload.get('description', 'unknown error')}"
            )
        return payload.get("result")

    async def send_document(
        self,
        chat_id: str,
        content: bytes,
        filename: str,
        caption: Optional[str] = None,
        message_thread_id: Optional[int] = None,
    ) -> Any:
        return await self._send_multipart(
            "sendDocument",
            "document",
            chat_id,
            content,
            filename,
            caption=caption,
            message_thread_id=message_thread_id,
        )

    async def send_photo(
        self,
        chat_id: str,
        content: bytes,
        filename: str,
        caption: Optional[str] = None,
        message_thread_id: Optional[int] = None,
    ) -> Any:
        return await self._send_multipart(
            "sendPhoto",
            "photo",
            chat_id,
            content,
            filename,
            caption=caption,
            message_thread_id=message_thread_id,
        )

    async def _send_multipart(
        self,
        method: str,
        field: str,
        chat_id: str,
        content: bytes,
        filename: str,
        caption: Optional[str] = None,
        message_thread_id: Optional[int] = None,
    ) -> Any:
        """File-carrying methods need multipart, unlike the JSON methods."""
        url = f"{_API_BASE}/bot{self._token}/{method}"
        data: dict[str, Any] = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
            data["parse_mode"] = "HTML"
        if message_thread_id is not None:
            data["message_thread_id"] = str(message_thread_id)
        resp = await self._http.post(url, data=data, files={field: (filename, content)})
        payload = resp.json()
        if not payload.get("ok"):
            raise TelegramAPIError(
                f"{method} failed: {payload.get('description', 'unknown error')}"
            )
        return payload.get("result")

    async def download_file(self, file_id: str) -> bytes:
        """Resolve a file_id via getFile and download its content."""
        info = await self.call("getFile", file_id=file_id)
        file_path = info.get("file_path") or ""
        url = f"{_API_BASE}/file/bot{self._token}/{file_path}"
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.content


def _drop_none(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}
