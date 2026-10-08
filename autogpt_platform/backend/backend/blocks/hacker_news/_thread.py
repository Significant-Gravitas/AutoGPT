"""Comment threads as flat lists, for the Hacker News Get Item block."""

from typing import Any, Callable

from ._models import (
    HackerNewsComment,
    HackerNewsItem,
    comment_from_node,
    item_from_firebase,
    item_from_thread,
    thread_node_from_firebase,
)

Node = dict[str, Any]
Read = tuple[HackerNewsItem, list[HackerNewsComment], int]


def read_thread(thread: Node, live: Node | None, limit: int) -> Read:
    """The item, up to `limit` of its comments, and its comment count, from an
    HN Search thread and `live`, the same item from HN's official API.

    HN Search keeps every comment's replies oldest first. The official API
    lists the item's direct replies (`kids`) in HN's ranking, so the direct
    replies are put in that order; deeper replies stay oldest first. HN Search
    also trails the live site by a minute or two, so the points and comment
    count come from `live` when it has them. The comments still come from HN
    Search, so the newest can be missing from the list.
    """
    live = live or {}
    ranking = {kid: rank for rank, kid in enumerate(live.get("kids") or [])}
    replies = sorted(
        thread.get("children") or [],
        key=lambda node: ranking.get(node.get("id"), len(ranking)),
    )
    comments, found = flatten_thread(
        replies, lambda node: node.get("children") or [], limit
    )
    count = _comment_count(found, live)
    # The official API gives every job 1 point, which HN never shows.
    score = None if thread.get("type") == "job" else live.get("score")
    points = thread.get("points") if score is None else score
    return item_from_thread(thread, count, points), comments, count


def read_live(live: Node, replies: dict[int, Node], limit: int) -> Read:
    """The same from HN's official API, for an item HN Search doesn't have.

    `replies` holds the comments fetched below the item, by id. The official
    API lists replies in HN's ranking at every level.
    """
    nodes = {reply_id: thread_node_from_firebase(r) for reply_id, r in replies.items()}

    def children(node: Node) -> list[Node]:
        return [nodes[kid] for kid in node["kids"] if kid in nodes]

    roots = [nodes[kid] for kid in live.get("kids") or [] if kid in nodes]
    comments, found = flatten_thread(roots, children, limit)
    return item_from_firebase(live), comments, _comment_count(found, live)


def flatten_thread(
    roots: list[Node], children: Callable[[Node], list[Node]], limit: int
) -> tuple[list[HackerNewsComment], int]:
    """Comments depth first, each followed by its replies, and how many there are.

    A comment with no author or text (deleted or dead) is left out, but its
    replies are kept. Only the first `limit` comments are returned; the count
    covers them all. A comment that appears twice is only read once.
    """
    comments: list[HackerNewsComment] = []
    count = 0
    seen: set[Any] = set()
    stack = [(node, 1) for node in reversed(roots)]
    while stack:
        node, depth = stack.pop()
        if node.get("id") in seen:
            continue
        seen.add(node.get("id"))
        stack.extend((child, depth + 1) for child in reversed(children(node)))
        if not (node.get("author") and node.get("text")):
            continue
        count += 1
        if len(comments) < limit:
            comments.append(comment_from_node(node, depth))
    return comments, count


def _comment_count(found: int, live: Node) -> int:
    """HN's own count (`descendants`) when it is higher than the comments
    found, which can miss the newest ones or the ones not fetched."""
    return max(found, live.get("descendants") or 0)
