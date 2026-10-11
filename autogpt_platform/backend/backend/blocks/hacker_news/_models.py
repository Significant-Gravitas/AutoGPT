"""Output models for the Hacker News blocks, and how each API's JSON maps to them.

HN Search search hits, HN Search items (/items/<id>) and items from HN's
official API name the same things differently (`author` or `by`, `points` or
`score`, `created_at_i` or `time`), so each has its own mapping.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ._api import item_url
from ._html import html_to_text

# HN Search lists the item type among a hit's _tags, next to tags such as
# "author_pg", "story_8863" and "show_hn".
_ITEM_TYPES = ("story", "comment", "poll", "pollopt", "job")


class HackerNewsItem(BaseModel):
    """A Hacker News story, comment, poll or job."""

    id: int = Field(description="Hacker News item id")
    type: str = Field(description="story, comment, poll, pollopt or job")
    title: str | None = Field(
        default=None,
        description="Title. For a comment found by search, the title of its story.",
    )
    url: str | None = Field(
        default=None,
        description=(
            "The link a story points to. Empty for text posts such as Ask HN, "
            "and for comments."
        ),
    )
    text: str | None = Field(
        default=None,
        description="Plain text of a comment, or of a text post such as Ask HN",
    )
    author: str | None = Field(default=None, description="Username of the poster")
    points: int | None = Field(
        default=None, description="Points (upvotes). Only stories and polls have them."
    )
    num_comments: int | None = Field(
        default=None, description="Number of comments on a story or poll"
    )
    created_at: str | None = Field(
        default=None, description="When it was posted, in ISO 8601 (UTC)"
    )
    hn_url: str = Field(description="Link to the item on Hacker News")
    story_id: int | None = Field(
        default=None, description="Id of the story it belongs to, when known"
    )
    parent_id: int | None = Field(
        default=None, description="For a comment, the id of the item it replies to"
    )


class HackerNewsStory(BaseModel):
    """A story on one of Hacker News's lists, with its place on the list."""

    id: int = Field(description="Hacker News item id")
    rank: int = Field(description="Position on the list, starting at 1")
    type: str = Field(description="story, poll or job")
    title: str | None = Field(default=None, description="Title")
    url: str | None = Field(
        default=None,
        description="The link the story points to. Empty for text posts such as Ask HN.",
    )
    text: str | None = Field(
        default=None, description="Plain text of a text post such as Ask HN"
    )
    author: str | None = Field(default=None, description="Username of the poster")
    points: int | None = Field(
        default=None, description="Points (upvotes). Jobs have none."
    )
    num_comments: int | None = Field(
        default=None, description="Number of comments. Jobs have none."
    )
    created_at: str | None = Field(
        default=None, description="When it was posted, in ISO 8601 (UTC)"
    )
    hn_url: str = Field(description="Link to the story on Hacker News")


class HackerNewsComment(BaseModel):
    """A comment in a thread, with how deep it is nested."""

    id: int = Field(description="Hacker News item id of the comment")
    author: str = Field(description="Username of the commenter")
    text: str = Field(description="Plain text of the comment")
    created_at: str | None = Field(
        default=None, description="When it was posted, in ISO 8601 (UTC)"
    )
    parent_id: int | None = Field(
        default=None, description="Id of the story or comment it replies to"
    )
    depth: int = Field(description="Nesting level: 1 is a direct reply to the item")
    hn_url: str = Field(description="Link to the comment on Hacker News")


def item_from_hit(hit: dict[str, Any]) -> HackerNewsItem:
    """An HN Search hit: a story, comment, poll or job."""
    tags = hit.get("_tags") or []
    item_type = next((tag for tag in tags if tag in _ITEM_TYPES), "story")
    is_comment = item_type == "comment"
    item_id = int(hit["objectID"])
    return HackerNewsItem(
        id=item_id,
        type=item_type,
        title=(hit.get("story_title") if is_comment else hit.get("title")) or None,
        url=None if is_comment else hit.get("url") or None,
        text=html_to_text(hit.get("comment_text") or hit.get("story_text")) or None,
        author=hit.get("author") or None,
        points=hit.get("points"),
        num_comments=hit.get("num_comments"),
        created_at=iso_time(hit.get("created_at_i")),
        hn_url=item_url(item_id),
        story_id=hit.get("story_id"),
        parent_id=hit.get("parent_id"),
    )


def item_from_thread(
    node: dict[str, Any], comment_count: int, points: int | None
) -> HackerNewsItem:
    """The item at the top of an HN Search thread (the /items/<id> endpoint).

    That endpoint has no comment count, and its points trail the live site,
    so the caller supplies both.
    """
    item_id = node["id"]
    item_type = node.get("type") or "story"
    return HackerNewsItem(
        id=item_id,
        type=item_type,
        title=node.get("title") or None,
        url=node.get("url") or None,
        text=html_to_text(node.get("text")) or None,
        author=node.get("author") or None,
        points=points,
        num_comments=comment_count if item_type in ("story", "poll") else None,
        created_at=iso_time(node.get("created_at_i")),
        hn_url=item_url(item_id),
        story_id=node.get("story_id"),
        parent_id=node.get("parent_id"),
    )


def item_from_firebase(item: dict[str, Any]) -> HackerNewsItem:
    """An item from HN's official API."""
    item_id = item["id"]
    item_type = item.get("type") or "story"
    return HackerNewsItem(
        id=item_id,
        type=item_type,
        title=item.get("title") or None,
        url=item.get("url") or None,
        text=html_to_text(item.get("text")) or None,
        author=item.get("by") or None,
        # The API gives every job 1 point, which HN never shows.
        points=None if item_type == "job" else item.get("score"),
        num_comments=item.get("descendants"),
        created_at=iso_time(item.get("time")),
        hn_url=item_url(item_id),
        story_id=item_id if item_type in ("story", "poll") else None,
        parent_id=item.get("parent"),
    )


def story_from_firebase(item: dict[str, Any], rank: int) -> HackerNewsStory:
    """An item from one of HN's story lists, at `rank` on the list."""
    fields = item_from_firebase(item).model_dump(exclude={"story_id", "parent_id"})
    return HackerNewsStory(rank=rank, **fields)


def comment_from_node(node: dict[str, Any], depth: int) -> HackerNewsComment:
    """A comment from an HN Search thread, `depth` levels below the item."""
    return HackerNewsComment(
        id=node["id"],
        author=node["author"],
        text=html_to_text(node["text"]),
        created_at=iso_time(node.get("created_at_i")),
        parent_id=node.get("parent_id"),
        depth=depth,
        hn_url=item_url(node["id"]),
    )


def thread_node_from_firebase(item: dict[str, Any]) -> dict[str, Any]:
    """A comment from HN's official API, shaped like a comment in an HN Search
    thread. A dead or deleted comment gets no author or text, as HN Search
    shows it."""
    hidden = item.get("deleted") or item.get("dead")
    return {
        "id": item["id"],
        "author": None if hidden else item.get("by"),
        "text": None if hidden else item.get("text"),
        "created_at_i": item.get("time"),
        "parent_id": item.get("parent"),
        "kids": item.get("kids") or [],
    }


def iso_time(timestamp: int | None) -> str | None:
    """Unix seconds as ISO 8601 in UTC, e.g. 2026-10-05T14:44:04Z."""
    if timestamp is None:
        return None
    moment = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")
