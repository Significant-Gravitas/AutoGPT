from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from x_twitter_scraper.types.shared import PaginatedTweets, SearchTweet

from backend.blocks.xquik._config import xquik
from backend.blocks.xquik.search import (
    XquikSearchTweetsBlock,
    XquikSortOrder,
    XquikTweetResult,
)
from backend.data.execution import ExecutionContext
from backend.data.model import APIKeyCredentials
from backend.util.exceptions import BlockUnknownError


def _page(
    *, with_author: bool = True, next_cursor: str = "next-page"
) -> PaginatedTweets:
    payload = {
        "id": "1991000000000000000",
        "text": "AutoGPT workflow example",
        "url": "https://x.com/example/status/1991000000000000000",
        "createdAt": "2026-08-20T12:00:00.000Z",
        "bookmarkCount": 1,
        "likeCount": 5,
        "quoteCount": 2,
        "replyCount": 3,
        "retweetCount": 4,
        "viewCount": 100,
    }
    if with_author:
        payload["author"] = {
            "id": "42",
            "name": "Auto-GPT",
            "username": "Auto_GPT",
        }
    tweet = SearchTweet.model_validate(payload)
    return PaginatedTweets(
        tweets=[tweet],
        has_next_page=bool(next_cursor),
        next_cursor=next_cursor,
    )


def _credentials() -> APIKeyCredentials:
    credentials = xquik.get_test_credentials()
    assert isinstance(credentials, APIKeyCredentials)
    return credentials


@pytest.mark.asyncio
async def test_search_maps_filters_and_yields_page_outputs():
    block = XquikSearchTweetsBlock()
    credentials = _credentials()
    input_data = block.Input(
        credentials=xquik.get_test_credentials().model_dump(),
        query="autonomous agents",
        limit=25,
        sort_order=XquikSortOrder.TOP,
        language="en",
        from_user="Auto_GPT",
        since_date=date(2026, 1, 1),
        until_date=date(2026, 8, 1),
        cursor="current-page",
    )

    with patch.object(block, "_search", new=AsyncMock(return_value=_page())) as search:
        outputs = [
            output async for output in block.run(input_data, credentials=credentials)
        ]

    search.assert_awaited_once_with(
        credentials,
        q="autonomous agents",
        limit=25,
        query_type="Top",
        language="en",
        from_user="Auto_GPT",
        since_date=date(2026, 1, 1),
        until_date=date(2026, 8, 1),
        cursor="current-page",
    )
    tweet = XquikTweetResult(
        id="1991000000000000000",
        text="AutoGPT workflow example",
        url="https://x.com/example/status/1991000000000000000",
        created_at="2026-08-20T12:00:00.000Z",
        author_id="42",
        author_name="Auto-GPT",
        author_username="Auto_GPT",
        bookmark_count=1,
        like_count=5,
        quote_count=2,
        reply_count=3,
        retweet_count=4,
        view_count=100,
    )
    assert outputs == [
        ("tweets", [tweet]),
        ("tweet", tweet),
        ("next_cursor", "next-page"),
        ("has_next_page", True),
    ]


@pytest.mark.asyncio
async def test_search_omits_empty_optional_filters():
    block = XquikSearchTweetsBlock()
    credentials = _credentials()
    input_data = block.Input(
        credentials=xquik.get_test_credentials().model_dump(),
        query="AutoGPT",
    )

    with patch.object(block, "_search", new=AsyncMock(return_value=_page())) as search:
        async for _ in block.run(input_data, credentials=credentials):
            pass

    search.assert_awaited_once_with(
        credentials,
        q="AutoGPT",
        limit=20,
        query_type="Latest",
    )


@pytest.mark.asyncio
async def test_search_omits_empty_cursor_output_and_tolerates_missing_author():
    block = XquikSearchTweetsBlock()
    credentials = _credentials()
    input_data = block.Input(
        credentials=xquik.get_test_credentials().model_dump(),
        query="AutoGPT",
    )

    with patch.object(
        block,
        "_search",
        new=AsyncMock(return_value=_page(with_author=False, next_cursor="")),
    ):
        outputs = [
            output async for output in block.run(input_data, credentials=credentials)
        ]

    assert [name for name, _ in outputs] == ["tweets", "tweet", "has_next_page"]
    tweet = outputs[1][1]
    assert isinstance(tweet, XquikTweetResult)
    assert tweet.author_id is None
    assert tweet.author_name is None
    assert tweet.author_username is None
    assert outputs[-1] == ("has_next_page", False)


@pytest.mark.asyncio
async def test_executor_wraps_sdk_errors_with_block_context():
    block = XquikSearchTweetsBlock()
    credentials = _credentials()
    input_data = block.Input(
        credentials=xquik.get_test_credentials().model_dump(),
        query="AutoGPT",
    )

    with (
        patch.object(
            block,
            "_search",
            new=AsyncMock(side_effect=RuntimeError("upstream unavailable")),
        ),
        pytest.raises(BlockUnknownError, match="upstream unavailable") as error,
    ):
        async for _ in block.execute(
            input_data.model_dump(),
            execution_context=ExecutionContext(),
            credentials=credentials,
        ):
            pass

    assert (error.value.block_name, error.value.block_id) == (block.name, block.id)


@pytest.mark.asyncio
async def test_sdk_client_receives_the_secret_and_search_arguments():
    block = XquikSearchTweetsBlock()
    credentials = _credentials()
    client = MagicMock()
    client.x.tweets.search = AsyncMock(return_value=_page())
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "backend.blocks.xquik.search.AsyncXTwitterScraper",
        return_value=context,
    ) as client_factory:
        page = await block._search(credentials, q="AutoGPT", limit=10)

    client_factory.assert_called_once_with(api_key="mock-xquik-api-key")
    client.x.tweets.search.assert_awaited_once_with(q="AutoGPT", limit=10)
    assert page == _page()


def test_output_schema_is_acyclic_for_autogpt_validation():
    properties = XquikSearchTweetsBlock.Output.jsonschema()["properties"]

    assert set(properties) == {
        "error",
        "tweets",
        "tweet",
        "next_cursor",
        "has_next_page",
    }


def test_provider_declares_api_key_auth():
    assert xquik.supported_auth_types == {"api_key"}
    assert xquik.description == "Public X post search without an X developer account"
