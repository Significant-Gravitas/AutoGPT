"""
Slack Web API helper functions.

Provides utilities for making authenticated requests to the Slack Web API.
All Slack API methods are POST requests to https://slack.com/api/{method}
with a Bearer token in the Authorization header.
"""

import logging
from typing import Any, Optional

from pydantic import BaseModel

from backend.data.model import APIKeyCredentials
from backend.util.request import Requests

logger = logging.getLogger(__name__)

SLACK_API_BASE = "https://slack.com/api"


class SlackMessageResult(BaseModel, extra="allow"):
    ts: str
    channel: str
    message: dict[str, Any] = {}


class SlackAPIException(ValueError):
    """Exception raised for Slack API errors."""

    def __init__(self, error: str):
        super().__init__(f"Slack API error: {error}")
        self.error = error


async def call_slack_api(
    credentials: APIKeyCredentials,
    method: str,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Make an authenticated POST request to the Slack Web API.

    Args:
        credentials: Slack bot token credentials
        method: API method name (e.g., "chat.postMessage")
        data: JSON body parameters

    Returns:
        Full API response dict (ok=True guaranteed)

    Raises:
        SlackAPIException: If Slack returns ok=false
    """
    token = credentials.api_key.get_secret_value()
    url = f"{SLACK_API_BASE}/{method}"
    headers = {"Authorization": f"Bearer {token}"}

    response = await Requests().post(url, json=data or {}, headers=headers)
    result: dict[str, Any] = response.json()

    if not result.get("ok"):
        raise SlackAPIException(result.get("error", "unknown_error"))

    return result


async def post_message(
    credentials: APIKeyCredentials,
    channel: str,
    text: str,
    thread_ts: Optional[str] = None,
    username: Optional[str] = None,
    icon_emoji: Optional[str] = None,
    unfurl_links: bool = True,
    mrkdwn: bool = True,
) -> SlackMessageResult:
    """Post a message to a Slack channel or DM."""
    data: dict[str, Any] = {
        "channel": channel,
        "text": text,
        "mrkdwn": mrkdwn,
        "unfurl_links": unfurl_links,
    }
    if thread_ts:
        data["thread_ts"] = thread_ts
    if username:
        data["username"] = username
    if icon_emoji:
        data["icon_emoji"] = icon_emoji

    result = await call_slack_api(credentials, "chat.postMessage", data)
    return SlackMessageResult(
        ts=result.get("ts", ""),
        channel=result.get("channel", ""),
        message=result.get("message", {}),
    )
