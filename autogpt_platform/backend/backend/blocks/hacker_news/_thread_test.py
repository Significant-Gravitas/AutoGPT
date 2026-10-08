"""Unit tests for flattening Hacker News comment threads into reading order."""

from typing import Any

from backend.blocks.hacker_news._thread import flatten_thread, read_live, read_thread


def comment(
    comment_id: int, parent_id: int, *replies: dict[str, Any], deleted: bool = False
) -> dict[str, Any]:
    """A comment as HN Search returns it inside a thread."""
    return {
        "id": comment_id,
        "author": None if deleted else f"user{comment_id}",
        "text": None if deleted else f"Comment {comment_id}",
        "parent_id": parent_id,
        "created_at_i": 1790780400 + comment_id,
        "children": list(replies),
    }


def story(*replies: dict[str, Any]) -> dict[str, Any]:
    return {"id": 1, "type": "story", "title": "A story", "children": list(replies)}


def children(node: dict[str, Any]) -> list[dict[str, Any]]:
    return node.get("children") or []


def ids_and_depths(comments) -> list[tuple[int, int]]:
    return [(c.id, c.depth) for c in comments]


def test_flatten_puts_each_comment_before_its_replies():
    thread = [
        comment(10, 1, comment(11, 10, comment(111, 11)), comment(12, 10)),
        comment(20, 1),
    ]
    comments, count = flatten_thread(thread, children, limit=100)
    assert ids_and_depths(comments) == [(10, 1), (11, 2), (111, 3), (12, 2), (20, 1)]
    assert count == 5
    assert comments[2].parent_id == 11
    assert comments[2].text == "Comment 111"
    assert comments[2].hn_url == "https://news.ycombinator.com/item?id=111"


def test_flatten_leaves_out_deleted_comments_but_keeps_their_replies():
    thread = [comment(10, 1, comment(11, 10), deleted=True), comment(20, 1)]
    comments, count = flatten_thread(thread, children, limit=100)
    assert ids_and_depths(comments) == [(11, 2), (20, 1)]
    assert comments[0].parent_id == 10
    assert count == 2


def test_flatten_cuts_the_list_but_counts_every_comment():
    thread = [comment(10, 1, comment(11, 10)), comment(20, 1), comment(30, 1)]
    comments, count = flatten_thread(thread, children, limit=2)
    assert ids_and_depths(comments) == [(10, 1), (11, 2)]
    assert count == 4
    assert flatten_thread(thread, children, limit=0) == ([], 4)


def test_flatten_handles_threads_deeper_than_the_recursion_limit():
    deepest = comment(5000, 4999)
    node = deepest
    for comment_id in range(4999, 9, -1):
        node = comment(comment_id, comment_id - 1, node)
    comments, count = flatten_thread([node], children, limit=10_000)
    assert count == 4991
    assert comments[-1].depth == 4991


def test_thread_replies_follow_hn_ranking_and_deeper_ones_stay_oldest_first():
    thread = story(
        comment(10, 1, comment(11, 10), comment(12, 10)),
        comment(20, 1),
        comment(30, 1),
        comment(40, 1),  # newer than the live item: not ranked yet
    )
    live = {"id": 1, "kids": [30, 10, 99, 20]}
    item, comments, count = read_thread(thread, live, limit=100)
    assert ids_and_depths(comments) == [
        (30, 1),
        (10, 1),
        (11, 2),
        (12, 2),
        (20, 1),
        (40, 1),
    ]
    assert count == 6
    assert (item.id, item.num_comments) == (1, 6)


def test_thread_without_a_live_item_stays_oldest_first():
    thread = story(comment(10, 1), comment(20, 1))
    _, comments, _ = read_thread(thread, None, limit=100)
    assert ids_and_depths(comments) == [(10, 1), (20, 1)]


def test_thread_with_no_comments():
    item, comments, count = read_thread(story(), {"id": 1}, limit=100)
    assert (comments, count, item.num_comments) == ([], 0, 0)


def test_thread_points_and_count_come_from_the_live_item():
    # HN Search trails the live site: it has 2 of the story's 5 comments.
    thread = {**story(comment(10, 1), comment(20, 1)), "points": 10}
    live = {"id": 1, "type": "story", "score": 15, "descendants": 5, "kids": [20, 10]}
    item, comments, count = read_thread(thread, live, limit=100)
    assert (item.points, item.num_comments, count) == (15, 5, 5)
    assert ids_and_depths(comments) == [(20, 1), (10, 1)]


def test_thread_count_is_never_below_the_comments_found():
    thread = story(comment(10, 1), comment(20, 1), comment(30, 1))
    live = {"id": 1, "type": "story", "score": 3, "descendants": 2}
    item, _, count = read_thread(thread, live, limit=100)
    assert (item.num_comments, count) == (3, 3)


def test_thread_without_a_live_item_uses_hn_search_points_and_count():
    thread = {**story(comment(10, 1)), "points": 10}
    item, _, count = read_thread(thread, None, limit=100)
    assert (item.points, item.num_comments, count) == (10, 1, 1)


def test_thread_job_shows_no_points_even_from_the_live_item():
    job = {"id": 2, "type": "job", "title": "Acme is hiring", "points": None}
    item, _, count = read_thread(job, {"id": 2, "type": "job", "score": 1}, limit=10)
    assert (item.points, item.num_comments, count) == (None, None, 0)


def test_thread_for_a_comment_counts_its_replies():
    thread = {
        **comment(10, 1, comment(11, 10), comment(12, 10)),
        "type": "comment",
        "story_id": 1,
    }
    live = {"id": 10, "type": "comment", "by": "user10", "parent": 1, "kids": [12, 11]}
    item, comments, count = read_thread(thread, live, limit=100)
    assert (item.type, item.points, item.num_comments) == ("comment", None, None)
    assert (item.story_id, item.parent_id) == (1, 1)
    assert ids_and_depths(comments) == [(12, 1), (11, 1)]
    assert count == 2


def firebase(item_id: int, parent: int, *kids: int, **extra: Any) -> dict[str, Any]:
    """A comment as HN's official API returns it."""
    return {
        "id": item_id,
        "by": f"user{item_id}",
        "text": f"Comment {item_id}",
        "time": 1790780400 + item_id,
        "parent": parent,
        "kids": list(kids),
        "type": "comment",
        **extra,
    }


def test_live_comments_follow_hn_ranking_at_every_level():
    live = {"id": 1, "type": "story", "by": "op", "kids": [20, 10], "descendants": 4}
    replies = {
        10: firebase(10, 1, 12, 11),
        11: firebase(11, 10),
        12: firebase(12, 10),
        20: firebase(20, 1),
    }
    item, comments, count = read_live(live, replies, limit=100)
    assert ids_and_depths(comments) == [(20, 1), (10, 1), (12, 2), (11, 2)]
    assert comments[1].author == "user10"
    assert count == 4
    assert (item.id, item.author, item.num_comments) == (1, "op", 4)


def test_live_comments_skip_dead_deleted_and_unfetched_ones():
    live = {"id": 1, "type": "story", "kids": [10, 20, 30, 40]}
    replies = {
        10: firebase(10, 1, 11, dead=True),
        11: firebase(11, 10),
        20: {"id": 20, "deleted": True, "parent": 1, "type": "comment"},
        30: firebase(30, 1),
        # 40 wasn't fetched (over the limit, or gone)
    }
    _, comments, count = read_live(live, replies, limit=100)
    assert ids_and_depths(comments) == [(11, 2), (30, 1)]
    assert count == 2


def test_live_count_uses_hn_count_when_not_every_reply_was_fetched():
    live = {"id": 1, "type": "story", "kids": [10, 20], "descendants": 57}
    replies = {10: firebase(10, 1), 20: firebase(20, 1)}
    _, comments, count = read_live(live, replies, limit=1)
    assert ids_and_depths(comments) == [(10, 1)]
    assert count == 57


def test_live_comments_that_list_each_other_are_read_once():
    # Malformed data: 10 and 11 reply to each other, and 10 is listed twice.
    live = {"id": 1, "type": "story", "kids": [10, 10]}
    replies = {10: firebase(10, 1, 11), 11: firebase(11, 10, 10)}
    _, comments, count = read_live(live, replies, limit=100)
    assert ids_and_depths(comments) == [(10, 1), (11, 2)]
    assert count == 2


def test_live_item_without_replies():
    live = {"id": 5, "type": "comment", "by": "x", "text": "hi", "parent": 1}
    item, comments, count = read_live(live, {}, limit=100)
    assert (item.id, item.parent_id, comments, count) == (5, 1, [], 0)
