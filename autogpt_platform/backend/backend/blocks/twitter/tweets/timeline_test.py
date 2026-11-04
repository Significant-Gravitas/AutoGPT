import pytest
import tweepy
from tweepy.client import Response

from backend.blocks.twitter._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.twitter.tweets.timeline import (
    TwitterGetHomeTimelineBlock,
    TwitterGetUserMentionsBlock,
    TwitterGetUserTweetsBlock,
)

# test_block.py skips these blocks (they need a live account), so this is their only run.
BLOCKS = [
    (TwitterGetUserMentionsBlock, "get_users_mentions", {"id": "12345"}),
    (TwitterGetHomeTimelineBlock, "get_home_timeline", {}),
    (TwitterGetUserTweetsBlock, "get_users_tweets", {"id": "12345"}),
]

FULL_RESPONSE = Response(
    data=[
        tweepy.Tweet({"id": "11", "text": "one", "edit_history_tweet_ids": ["11"]}),
        tweepy.Tweet({"id": "12", "text": "two", "edit_history_tweet_ids": ["12"]}),
    ],
    includes={
        "users": [
            tweepy.User({"id": "21", "name": "A", "username": "a"}),
            tweepy.User({"id": "22", "name": "B", "username": "b"}),
        ]
    },
    errors=[],
    meta={"next_token": "nt", "result_count": 2},
)


@pytest.fixture
def fake_client(monkeypatch):
    calls: list[tuple[str, dict]] = []
    answer = {"response": FULL_RESPONSE}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def _answer(self, endpoint: str, params: dict):
            calls.append((endpoint, params))
            return answer["response"]

        def get_users_mentions(self, **params):
            return self._answer("get_users_mentions", params)

        def get_home_timeline(self, **params):
            return self._answer("get_home_timeline", params)

        def get_users_tweets(self, **params):
            return self._answer("get_users_tweets", params)

    monkeypatch.setattr(tweepy, "Client", FakeClient)
    return calls, answer


async def _run(block_cls, **inputs):
    raw = {"credentials": TEST_CREDENTIALS_INPUT, **inputs}
    if "user_id" in block_cls.Input.model_fields:
        raw["user_id"] = "12345"
    return [
        output
        async for output in block_cls().run(
            block_cls.Input(**raw), credentials=TEST_CREDENTIALS
        )
    ]


@pytest.mark.parametrize("block_cls, endpoint, id_param", BLOCKS)
async def test_each_block_calls_its_own_endpoint(
    fake_client, block_cls, endpoint, id_param
):
    calls, _ = fake_client

    await _run(block_cls, pagination_token="tok")

    assert calls == [
        (
            endpoint,
            {
                **id_param,
                "max_results": 10,
                "pagination_token": "tok",
                "user_auth": False,
            },
        )
    ]


@pytest.mark.parametrize("block_cls, endpoint, id_param", BLOCKS)
async def test_outputs_follow_the_response(fake_client, block_cls, endpoint, id_param):
    outputs = await _run(block_cls)

    assert [name for name, _ in outputs] == [
        "ids",
        "texts",
        "userIds",
        "userNames",
        "next_token",
        "data",
        "included",
        "meta",
    ]
    values = dict(outputs)
    assert values["ids"] == ["11", "12"]
    assert values["texts"] == ["one", "two"]
    assert values["userIds"] == ["21", "22"]
    assert values["userNames"] == ["a", "b"]
    assert values["next_token"] == "nt"
    assert values["meta"] == {"next_token": "nt", "result_count": 2}


@pytest.mark.parametrize("block_cls, endpoint, id_param", BLOCKS)
async def test_an_empty_response_yields_an_error(
    fake_client, block_cls, endpoint, id_param
):
    _, answer = fake_client
    answer["response"] = Response(data=None, includes={}, errors=[], meta={})

    assert await _run(block_cls) == [("error", "Unexpected error: No tweets found")]
