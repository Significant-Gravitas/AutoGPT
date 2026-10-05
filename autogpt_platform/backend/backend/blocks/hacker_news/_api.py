"""Requests to the two public Hacker News APIs the blocks use.

HN Search (hn.algolia.com, run by Algolia) answers searches, and returns an
item with its whole comment thread in one call. HN's official API on Firebase
(hacker-news.firebaseio.com) has the live story lists, single items and user
profiles. Neither needs an account or a key.
"""

import asyncio
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from backend.util.request import Requests, Response

ALGOLIA_URL = "https://hn.algolia.com/api/v1"
FIREBASE_URL = "https://hacker-news.firebaseio.com/v0"
HN_URL = "https://news.ycombinator.com"

# The official API has one URL per item, so a list of stories takes one request
# per story. This many run at once.
MAX_CONCURRENT_REQUESTS = 10

_TRUSTED_ORIGINS = ["https://hn.algolia.com", "https://hacker-news.firebaseio.com"]
_DIGITS = re.compile(r"[0-9]+")
# HN usernames are letters, digits, - and _. New accounts get 2 to 15 of them,
# but older ones predate that rule, so the length check is deliberately loose:
# a name HN doesn't know fails as "not found" instead.
_USERNAME = re.compile(r"[A-Za-z0-9_-]{1,32}")


class HackerNewsError(Exception):
    """A Hacker News API failure, with a message the user can act on."""


async def search(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
    """Run an HN Search query; `endpoint` is "search" or "search_by_date"."""
    return await get_json(f"{ALGOLIA_URL}/{endpoint}", params) or {}


async def get_thread(item_id: int) -> dict[str, Any] | None:
    """An item and its whole comment tree from HN Search.

    None when HN Search doesn't have the item: there is no such item, it is
    dead, or it was posted in the last few minutes and isn't indexed yet.
    """
    return await get_json(f"{ALGOLIA_URL}/items/{item_id}")


async def get_item(item_id: int) -> dict[str, Any] | None:
    """An item from HN's official API, or None when there is no such item."""
    return await get_json(f"{FIREBASE_URL}/item/{item_id}.json")


async def get_items(item_ids: list[int]) -> list[dict[str, Any] | None]:
    """Items from HN's official API, in the order of `item_ids`."""
    slots = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def fetch(item_id: int) -> dict[str, Any] | None:
        async with slots:
            return await get_item(item_id)

    return list(await asyncio.gather(*(fetch(item_id) for item_id in item_ids)))


async def get_replies(kids: list[int], limit: int) -> dict[int, dict[str, Any]]:
    """Up to `limit` comments below an item from HN's official API, by id.

    Fetched breadth first, so every comment's parent is fetched before it.
    Dead and deleted comments are fetched too, and count towards the limit.
    Each id is fetched once, even if the API lists it twice.
    """
    found: dict[int, dict[str, Any]] = {}
    queue = list(dict.fromkeys(kids))
    queued = set(queue)
    while queue and len(found) < limit:
        room = limit - len(found)
        batch, queue = queue[:room], queue[room:]
        for item in await get_items(batch):
            if item:
                found[item["id"]] = item
                new = [kid for kid in item.get("kids") or [] if kid not in queued]
                queued.update(new)
                queue.extend(new)
    return found


async def get_story_ids(list_name: str) -> list[int]:
    """The ids on one of HN's story lists, e.g. "topstories", in HN's order."""
    return await get_json(f"{FIREBASE_URL}/{list_name}.json") or []


async def get_user(username: str) -> dict[str, Any] | None:
    """A user's profile from HN's official API, or None when there is no such user."""
    return await get_json(f"{FIREBASE_URL}/user/{username}.json")


async def get_json(url: str, params: dict[str, str] | None = None) -> Any:
    """GET a Hacker News API URL and return its JSON, or None for a 404.

    The official API answers `null` rather than 404 for a missing item or
    user, which also comes back as None. 429 and 5xx responses are retried
    twice before giving up.
    """
    response = await Requests(
        trusted_origins=_TRUSTED_ORIGINS, raise_for_status=False, retry_max_attempts=3
    ).get(url, params=params)
    if response.status == 404:
        return None
    if not response.ok:
        raise api_error(url, response)
    return response.json()


def api_error(url: str, response: Response) -> HackerNewsError:
    """Turn a failed Hacker News API response into a message the user can act on."""
    if urlparse(url).hostname == "hn.algolia.com":
        api = "Hacker News search (hn.algolia.com)"
    else:
        api = "The Hacker News API"
    status = response.status
    if status == 429:
        return HackerNewsError(
            f"{api} is limiting how many requests it takes from this server "
            "(HTTP 429). Wait a few minutes, then try again."
        )
    if status >= 500:
        return HackerNewsError(
            f"{api} had a temporary problem (HTTP {status}). Try again in a minute."
        )
    body = response.json(fallback={}) if response.content else {}
    message = body.get("message") if isinstance(body, dict) else None
    detail = message or response.reason or "no details given"
    return HackerNewsError(f"{api} rejected the request (HTTP {status}): {detail}")


def parse_item_id(text: str) -> int | None:
    """The id in '8863' or in a link such as news.ycombinator.com/item?id=8863."""
    text = text.strip()
    if _DIGITS.fullmatch(text):
        return int(text)
    value = _link_id(text, "/item")
    return int(value) if value and _DIGITS.fullmatch(value) else None


def parse_username(text: str) -> str | None:
    """The username in 'pg' or in a link such as news.ycombinator.com/user?id=pg."""
    text = text.strip()
    if not _USERNAME.fullmatch(text):
        text = _link_id(text, "/user") or ""
    return text if _USERNAME.fullmatch(text) else None


def item_url(item_id: int) -> str:
    return f"{HN_URL}/item?id={item_id}"


def user_url(username: str) -> str:
    return f"{HN_URL}/user?id={username}"


def _link_id(text: str, path: str) -> str | None:
    """The id in a Hacker News link to `path`, written with or without https://."""
    try:
        url = urlparse(text if "://" in text else f"https://{text}")
        host = url.hostname
    except ValueError:  # e.g. an unclosed [ in the host
        return None
    if host != "news.ycombinator.com" or url.path != path:
        return None
    return next(iter(parse_qs(url.query).get("id", [])), None)
