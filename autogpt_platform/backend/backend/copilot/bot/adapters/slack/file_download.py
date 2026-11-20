"""Downloading the files a Slack message carries.

A file Slack hosts is private: fetching it takes the workspace's bot token as
a bearer credential. Not every file in an event is hosted by Slack, though. A
remote file (``files.remote.add``) carries, as its ``url_private``, the
``external_url`` chosen by the app that added it, and Slack relays that URL
unchanged inside the signed event.

So the token only ever goes to Slack's own file host, as Slack's guidance for
downloading files asks: remote files are refused outright, and any other
file's URL must be ``https`` on an exact host in ``config.FILE_HOSTS``. The
request goes through the platform's guarded client, which connects only to
addresses it has checked. Redirects are not followed, and the body is read
under the size cap whatever size the event declared.
"""

from typing import Any

import aiohttp

from backend.util.request import Requests, parse_url

from . import config

# Bounds connecting and each read rather than the whole transfer, so a large
# file on a slow link still completes.
_TIMEOUT = aiohttp.ClientTimeout(connect=30, sock_read=30)


async def download(file_obj: dict[str, Any], token: str, *, max_bytes: int) -> bytes:
    """Download a Slack-hosted file with the workspace's bot ``token``.

    Raises ``ValueError`` before anything is sent for a file Slack does not
    host, and for any response but a success: a redirect is refused rather
    than followed, since its target is unchecked.
    """
    url = _slack_file_url(file_obj)
    response = await Requests(
        raise_for_status=False,
        retry_max_attempts=1,
        max_response_bytes=max_bytes,
    ).get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        allow_redirects=False,
        timeout=_TIMEOUT,
    )
    if not response.ok:
        raise ValueError(f"Slack file download returned HTTP {response.status}")
    return response.content


def _slack_file_url(file_obj: dict[str, Any]) -> str:
    """The file's download URL, if the bot token may be sent to it."""
    if (
        file_obj.get("is_external")
        or file_obj.get("mode") == "external"
        or file_obj.get("filetype") == "remote"
    ):
        raise ValueError("remote Slack files are not downloaded")
    url = file_obj.get("url_private_download") or file_obj.get("url_private") or ""
    # Parsed as the request client parses it, so the host checked here is the
    # host it connects to (a backslash in the authority is read as a slash).
    parsed = parse_url(url)
    if parsed.scheme != "https" or parsed.hostname not in config.FILE_HOSTS:
        raise ValueError(f"not a Slack file URL (host {parsed.hostname!r})")
    return url
