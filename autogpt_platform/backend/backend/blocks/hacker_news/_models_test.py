"""Unit tests for mapping each Hacker News API's JSON to the blocks' models."""

import pytest

from backend.blocks.hacker_news._models import (
    HackerNewsItem,
    HackerNewsStory,
    iso_time,
    item_from_firebase,
    item_from_hit,
    item_from_thread,
    story_from_firebase,
    thread_node_from_firebase,
)


def test_story_hit():
    item = item_from_hit(
        {
            "_tags": ["story", "author_maker_jane", "story_49965697", "show_hn"],
            "objectID": "49965697",
            "title": "Show HN: Pocketlog, a tiny log viewer",
            "url": "https://github.com/example/pocketlog",
            "author": "maker_jane",
            "points": 1,
            "num_comments": 0,
            "created_at": "2026-10-05T14:56:05Z",
            "created_at_i": 1791212165,
            "story_id": 49965697,
        }
    )
    assert item == HackerNewsItem(
        id=49965697,
        type="story",
        title="Show HN: Pocketlog, a tiny log viewer",
        url="https://github.com/example/pocketlog",
        author="maker_jane",
        points=1,
        num_comments=0,
        created_at="2026-10-05T14:56:05Z",
        hn_url="https://news.ycombinator.com/item?id=49965697",
        story_id=49965697,
    )


def test_ask_hn_hit_has_text_and_no_link():
    item = item_from_hit(
        {
            "_tags": ["story", "author_someone", "story_1", "ask_hn"],
            "objectID": "1",
            "title": "Ask HN: Anyone else?",
            "story_text": "I wonder.<p>Do you?",
            "created_at_i": 1693926441,
        }
    )
    assert (item.url, item.text) == (None, "I wonder.\n\nDo you?")


def test_comment_hit_takes_its_story_title_but_not_its_link():
    item = item_from_hit(
        {
            "_tags": ["comment", "author_fan_club", "story_49917536"],
            "objectID": "49965549",
            "author": "fan_club",
            "comment_text": "Ours came with <i>no switch</i>.<p>I don&#x27;t know why.",
            "points": None,
            "story_id": 49917536,
            "story_title": "Why ceiling fans got so complicated",
            "story_url": "https://example.com/ceiling-fans",
            "parent_id": 49945418,
            "created_at_i": 1791211444,
        }
    )
    assert item == HackerNewsItem(
        id=49965549,
        type="comment",
        title="Why ceiling fans got so complicated",
        url=None,
        text="Ours came with *no switch*.\n\nI don't know why.",
        author="fan_club",
        created_at="2026-10-05T14:44:04Z",
        hn_url="https://news.ycombinator.com/item?id=49965549",
        story_id=49917536,
        parent_id=49945418,
    )


@pytest.mark.parametrize(
    "tags, expected",
    [
        (["poll", "author_x", "story_5"], "poll"),
        (["pollopt", "author_x"], "pollopt"),
        (["job", "author_x"], "job"),
        (["author_x", "comment", "story_5"], "comment"),
        ([], "story"),
    ],
)
def test_hit_type_comes_from_its_tags(tags: list[str], expected: str):
    assert item_from_hit({"_tags": tags, "objectID": "5"}).type == expected


def test_hit_with_missing_and_empty_fields():
    item = item_from_hit({"objectID": "7", "title": "", "url": "", "author": ""})
    assert item == HackerNewsItem(
        id=7, type="story", hn_url="https://news.ycombinator.com/item?id=7"
    )


def test_thread_item_takes_the_count_and_points_it_is_given():
    item = item_from_thread(
        {
            "id": 8863,
            "type": "story",
            "author": "dhouston",
            "title": "My YC app: Dropbox - Throw away your USB drive",
            "url": "http://www.getdropbox.com/u/2/screencast.html",
            "text": None,
            "points": 104,
            "parent_id": None,
            "story_id": 8863,
            "created_at": "2007-04-04T19:16:40.000Z",
            "created_at_i": 1175714200,
            "options": [],
            "children": [],
        },
        comment_count=71,
        points=110,
    )
    assert item == HackerNewsItem(
        id=8863,
        type="story",
        title="My YC app: Dropbox - Throw away your USB drive",
        url="http://www.getdropbox.com/u/2/screencast.html",
        author="dhouston",
        points=110,
        num_comments=71,
        created_at="2007-04-04T19:16:40Z",
        hn_url="https://news.ycombinator.com/item?id=8863",
        story_id=8863,
    )


def test_thread_comment_has_no_comment_count():
    item = item_from_thread(
        {
            "id": 8865,
            "type": "comment",
            "author": "dhouston",
            "text": "oh, and a mac port is coming :)",
            "points": None,
            "title": None,
            "url": None,
            "parent_id": 8863,
            "story_id": 8863,
            "created_at_i": 1175714575,
        },
        comment_count=3,
        points=None,
    )
    assert (item.num_comments, item.points, item.title) == (None, None, None)
    assert (item.story_id, item.parent_id) == (8863, 8863)
    assert item.text == "oh, and a mac port is coming :)"


def test_firebase_story():
    item = item_from_firebase(
        {
            "by": "ops_person",
            "descendants": 20,
            "id": 49964045,
            "kids": [49965311, 49964775],
            "score": 39,
            "text": "Our IP is on a blocklist.<p>It&#x27;s the whole ASN.",
            "time": 1791207720,
            "title": "Ask HN: Is your site blocked too?",
            "type": "story",
        }
    )
    assert item == HackerNewsItem(
        id=49964045,
        type="story",
        title="Ask HN: Is your site blocked too?",
        text="Our IP is on a blocklist.\n\nIt's the whole ASN.",
        author="ops_person",
        points=39,
        num_comments=20,
        created_at="2026-10-05T13:42:00Z",
        hn_url="https://news.ycombinator.com/item?id=49964045",
        story_id=49964045,
    )


def test_firebase_job_shows_no_points():
    item = item_from_firebase(
        {
            "by": "acme_hiring",
            "id": 49945904,
            "score": 1,
            "time": 1791046812,
            "title": "Acme (YC W24) Is Hiring",
            "type": "job",
            "url": "https://www.ycombinator.com/companies/acme/jobs/x",
        }
    )
    assert (item.type, item.points, item.num_comments, item.story_id) == (
        "job",
        None,
        None,
        None,
    )


def test_firebase_deleted_comment():
    item = item_from_firebase(
        {
            "deleted": True,
            "id": 11116487,
            "parent": 11116274,
            "time": 1455701075,
            "type": "comment",
        }
    )
    assert (item.author, item.text, item.parent_id, item.story_id) == (
        None,
        None,
        11116274,
        None,
    )


def test_story_from_a_list_keeps_its_rank():
    story = story_from_firebase(
        {
            "by": "dhouston",
            "descendants": 71,
            "id": 8863,
            "score": 104,
            "time": 1175714200,
            "title": "My YC app: Dropbox - Throw away your USB drive",
            "type": "story",
            "url": "http://www.getdropbox.com/u/2/screencast.html",
        },
        rank=4,
    )
    assert story == HackerNewsStory(
        id=8863,
        rank=4,
        type="story",
        title="My YC app: Dropbox - Throw away your USB drive",
        url="http://www.getdropbox.com/u/2/screencast.html",
        author="dhouston",
        points=104,
        num_comments=71,
        created_at="2007-04-04T19:16:40Z",
        hn_url="https://news.ycombinator.com/item?id=8863",
    )


@pytest.mark.parametrize(
    "item, author, text",
    [
        ({"id": 1, "by": "a", "text": "hi", "time": 5, "parent": 9}, "a", "hi"),
        ({"id": 1, "by": "a", "text": "", "dead": True, "parent": 9}, None, None),
        ({"id": 1, "deleted": True, "parent": 9, "kids": [2]}, None, None),
    ],
)
def test_firebase_comment_as_a_thread_node(item: dict, author: str, text: str):
    node = thread_node_from_firebase(item)
    assert (node["author"], node["text"], node["parent_id"]) == (author, text, 9)
    assert node["kids"] == item.get("kids", [])


@pytest.mark.parametrize(
    "timestamp, expected",
    [(0, "1970-01-01T00:00:00Z"), (1791211444, "2026-10-05T14:44:04Z"), (None, None)],
)
def test_iso_time(timestamp: int | None, expected: str | None):
    assert iso_time(timestamp) == expected
