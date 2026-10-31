import importlib
import inspect
from typing import ClassVar

import pytest
import tweepy
from tweepy.client import Response

from backend.blocks._base import Block
from backend.blocks.twitter._auth import TEST_CREDENTIALS

# test_block.py swaps these blocks' client calls for their test_mock, so this is the only run of them.
READS = {
    "lists.list_lookup.TwitterGetListBlock": "get_list",
    "lists.list_lookup.TwitterGetOwnedListsBlock": "get_owned_lists",
    "lists.list_members.TwitterGetListMembersBlock": "get_list_members",
    "lists.list_tweets_lookup.TwitterGetListTweetsBlock": "get_list_tweets",
    "lists.pinned_lists.TwitterGetPinnedListsBlock": "get_pinned_lists",
    "spaces.search_spaces.TwitterSearchSpacesBlock": "search_spaces",
    "spaces.spaces_lookup.TwitterGetSpacesBlock": "get_spaces",
    "spaces.spaces_lookup.TwitterGetSpaceByIdBlock": "get_space",
    "spaces.spaces_lookup.TwitterGetSpaceBuyersBlock": "get_space_buyers",
    "spaces.spaces_lookup.TwitterGetSpaceTweetsBlock": "get_space_tweets",
    "tweets.bookmark.TwitterGetBookmarkedTweetsBlock": "get_bookmarks",
    "tweets.like.TwitterGetLikingUsersBlock": "get_liking_users",
    "tweets.like.TwitterGetLikedTweetsBlock": "get_liked_tweets",
    "tweets.manage.TwitterSearchRecentTweetsBlock": "search_recent_tweets",
    "tweets.quote.TwitterGetQuoteTweetsBlock": "get_quote_tweets",
    "tweets.retweet.TwitterGetRetweetersBlock": "get_retweeters",
    "tweets.tweet_lookup.TwitterGetTweetBlock": "get_tweet",
    "tweets.tweet_lookup.TwitterGetTweetsBlock": "get_tweets",
    "users.blocks.TwitterGetBlockedUsersBlock": "get_blocked",
    "users.follows.TwitterGetFollowersBlock": "get_users_followers",
    "users.follows.TwitterGetFollowingBlock": "get_users_following",
    "users.mutes.TwitterGetMutedUsersBlock": "get_muted",
    "users.user_lookup.TwitterGetUserBlock": "get_user",
    "users.user_lookup.TwitterGetUsersBlock": "get_users",
}
WRITES = {
    "lists.list_follows.TwitterUnfollowListBlock": "unfollow_list",
    "lists.list_follows.TwitterFollowListBlock": "follow_list",
    "tweets.bookmark.TwitterBookmarkTweetBlock": "bookmark",
    "tweets.bookmark.TwitterRemoveBookmarkTweetBlock": "remove_bookmark",
    "tweets.hide.TwitterHideReplyBlock": "hide_reply",
    "tweets.hide.TwitterUnhideReplyBlock": "unhide_reply",
    "tweets.like.TwitterLikeTweetBlock": "like",
    "tweets.like.TwitterUnlikeTweetBlock": "unlike",
    "tweets.retweet.TwitterRetweetBlock": "retweet",
    "tweets.retweet.TwitterRemoveRetweetBlock": "unretweet",
    "users.follows.TwitterUnfollowUserBlock": "unfollow_user",
    "users.follows.TwitterFollowUserBlock": "follow_user",
    "users.mutes.TwitterUnmuteUserBlock": "unmute",
    "users.mutes.TwitterMuteUserBlock": "mute",
}

# Every field holds its own name, so an output's value says which field it came from.
# tweepy.List keeps its id as given and serves any other field from its payload.
RECORD = tweepy.List(
    {field: field for field in ("id", "name", "username", "text", "title")}
    | {"host_ids": ["host_ids"]}
)
OWNER = tweepy.List({field: f"owner.{field}" for field in ("id", "name", "username")})
# A lookup by one id answers with one record, every other endpoint with a list.
SINGLE_RECORD_ENDPOINTS = {"get_list", "get_space", "get_tweet", "get_user"}

# The answer field each output carries.
CARRIES = {
    "id": ("id", "ids", "list_ids", "tweet_ids", "buyer_ids", "user_ids"),
    "name": ("name", "name_", "names", "names_", "list_names"),
    "username": ("username", "username_", "usernames", "usernames_"),
    "text": ("text", "texts", "tweet_texts"),
    "title": ("title", "titles"),
    "host_ids": ("host_ids",),
    "owner.id": ("owner_id", "userId", "userIds"),
    "owner.username": ("owner_username", "userName", "userNames"),
    "next_token": ("next_token",),
}
FIELD_OF = {output: field for field, outputs in CARRIES.items() for output in outputs}
PASSED_THROUGH = {"data", "included", "includes", "meta"}


@pytest.fixture
def fake_client(monkeypatch):
    real_client = tweepy.Client

    class FakeClient:
        calls: ClassVar[list[str]] = []
        empty = False

        def __init__(self, **kwargs):
            pass

        def __getattr__(self, endpoint):
            def call(*args, **kwargs):
                # Binding to the real method refuses a parameter tweepy does not take.
                inspect.signature(getattr(real_client, endpoint)).bind(
                    self, *args, **kwargs
                )
                FakeClient.calls.append(endpoint)
                return _answer(endpoint, FakeClient.empty)

            return call

    monkeypatch.setattr(tweepy, "Client", FakeClient)
    return FakeClient


@pytest.mark.parametrize("path, endpoint", WRITES.items())
async def test_a_write_block_calls_its_endpoint(fake_client, path, endpoint):
    assert await _run(path) == [("success", True)]
    assert fake_client.calls == [endpoint]


@pytest.mark.parametrize("path, endpoint", READS.items())
async def test_a_read_block_yields_each_field_its_endpoint_answers(
    fake_client, path, endpoint
):
    outputs = await _run(path)

    assert fake_client.calls == [endpoint]
    declared = set(_block(path).Output.model_fields) - {"error"}
    assert sorted(name for name, _ in outputs) == sorted(declared)
    for name, value in outputs:
        if name not in PASSED_THROUGH:
            assert _flat(value) == [FIELD_OF[name]], name


@pytest.mark.parametrize("path, endpoint", READS.items())
async def test_a_read_block_reports_an_empty_answer_as_an_error(
    fake_client, path, endpoint
):
    fake_client.empty = True

    assert [name for name, _ in await _run(path)] == ["error"]


async def _run(path: str) -> list[tuple[str, object]]:
    block = _block(path)()
    inputs = block.input_schema(**block.test_input)
    return [output async for output in block.run(inputs, credentials=TEST_CREDENTIALS)]


def _block(path: str) -> type[Block]:
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(f"backend.blocks.twitter.{module}"), name)


def _answer(endpoint: str, empty: bool) -> Response:
    if empty:
        return Response(data=None, includes={}, errors=[], meta={})
    data = RECORD if endpoint in SINGLE_RECORD_ENDPOINTS else [RECORD]
    return Response(
        data=data,
        includes={"users": [OWNER]},
        errors=[],
        meta={"next_token": "next_token"},
    )


def _flat(value) -> list:
    if isinstance(value, list):
        return [leaf for item in value for leaf in _flat(item)]
    return [value]
