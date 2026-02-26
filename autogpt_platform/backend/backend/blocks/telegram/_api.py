"""
Telegram Bot API helper functions.

Provides utilities for making authenticated requests to the Telegram Bot API.
"""

import logging
from io import BytesIO
from typing import Any, Optional

from pydantic import BaseModel

from backend.data.model import APIKeyCredentials
from backend.util.request import Requests

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramMessageResult(BaseModel, extra="allow"):
    """Result from Telegram send/edit message API calls."""

    message_id: int = 0
    chat: dict[str, Any] = {}
    date: int = 0
    text: str = ""


class TelegramFileResult(BaseModel, extra="allow"):
    """Result from Telegram getFile API call."""

    file_id: str = ""
    file_unique_id: str = ""
    file_size: int = 0
    file_path: str = ""


class TelegramAPIException(ValueError):
    """Exception raised for Telegram API errors."""

    def __init__(self, message: str, error_code: int = 0):
        super().__init__(message)
        self.error_code = error_code


def get_bot_api_url(bot_token: str, method: str) -> str:
    """Construct Telegram Bot API URL for a method."""
    return f"{TELEGRAM_API_BASE}/bot{bot_token}/{method}"


def get_file_url(bot_token: str, file_path: str) -> str:
    """Construct Telegram file download URL."""
    return f"{TELEGRAM_API_BASE}/file/bot{bot_token}/{file_path}"


async def call_telegram_api(
    credentials: APIKeyCredentials,
    method: str,
    data: Optional[dict[str, Any]] = None,
) -> TelegramMessageResult:
    """
    Make a request to the Telegram Bot API.

    Args:
        credentials: Bot token credentials
        method: API method name (e.g., "sendMessage", "getFile")
        data: Request parameters

    Returns:
        API response result

    Raises:
        TelegramAPIException: If the API returns an error
    """
    token = credentials.api_key.get_secret_value()
    url = get_bot_api_url(token, method)

    response = await Requests().post(url, json=data or {})
    result = response.json()

    if not result.get("ok"):
        error_code = result.get("error_code", 0)
        description = result.get("description", "Unknown error")
        raise TelegramAPIException(description, error_code)

    return TelegramMessageResult(**result.get("result", {}))


async def call_telegram_api_with_file(
    credentials: APIKeyCredentials,
    method: str,
    file_field: str,
    file_data: bytes,
    filename: str,
    content_type: str,
    data: Optional[dict[str, Any]] = None,
) -> TelegramMessageResult:
    """
    Make a multipart/form-data request to the Telegram Bot API with a file upload.

    Args:
        credentials: Bot token credentials
        method: API method name (e.g., "sendPhoto", "sendVoice")
        file_field: Form field name for the file (e.g., "photo", "voice")
        file_data: Raw file bytes
        filename: Filename for the upload
        content_type: MIME type of the file
        data: Additional form parameters

    Returns:
        API response result

    Raises:
        TelegramAPIException: If the API returns an error
    """
    token = credentials.api_key.get_secret_value()
    url = get_bot_api_url(token, method)

    files = [(file_field, (filename, BytesIO(file_data), content_type))]

    response = await Requests().post(url, files=files, data=data or {})
    result = response.json()

    if not result.get("ok"):
        error_code = result.get("error_code", 0)
        description = result.get("description", "Unknown error")
        raise TelegramAPIException(description, error_code)

    return TelegramMessageResult(**result.get("result", {}))


async def get_file_info(
    credentials: APIKeyCredentials, file_id: str
) -> TelegramFileResult:
    """
    Get file information from Telegram.

    Args:
        credentials: Bot token credentials
        file_id: Telegram file_id from message

    Returns:
        File info dict containing file_id, file_unique_id, file_size, file_path
    """
    result = await call_telegram_api(credentials, "getFile", {"file_id": file_id})
    return TelegramFileResult(**result.model_dump())


async def get_file_download_url(credentials: APIKeyCredentials, file_id: str) -> str:
    """
    Get the download URL for a Telegram file.

    Args:
        credentials: Bot token credentials
        file_id: Telegram file_id from message

    Returns:
        Full download URL
    """
    token = credentials.api_key.get_secret_value()
    result = await get_file_info(credentials, file_id)
    file_path = result.file_path
    if not file_path:
        raise TelegramAPIException("No file_path returned from getFile")
    return get_file_url(token, file_path)


async def download_telegram_file(credentials: APIKeyCredentials, file_id: str) -> bytes:
    """
    Download a file from Telegram servers.

    Args:
        credentials: Bot token credentials
        file_id: Telegram file_id

    Returns:
        File content as bytes
    """
    url = await get_file_download_url(credentials, file_id)
    response = await Requests().get(url)
    return response.content
