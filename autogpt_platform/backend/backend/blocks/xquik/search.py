from datetime import date
from enum import Enum
from typing import Unpack

from x_twitter_scraper import AsyncXTwitterScraper
from x_twitter_scraper.types.shared import PaginatedTweets, SearchTweet
from x_twitter_scraper.types.x.tweet_search_params import TweetSearchParams
from x_twitter_scraper.types.x.tweet_search_response import TweetSearchResponse

from backend.sdk import (
    APIKeyCredentials,
    BaseModel,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import xquik


class XquikSortOrder(str, Enum):
    LATEST = "Latest"
    TOP = "Top"


class XquikTweetResult(BaseModel):
    id: str = SchemaField(description="X post ID")
    text: str = SchemaField(description="Post text")
    url: str | None = SchemaField(description="Post URL", default=None)
    created_at: str | None = SchemaField(description="Post creation time", default=None)
    language: str | None = SchemaField(description="Post language", default=None)
    author_id: str | None = SchemaField(description="Author ID", default=None)
    author_name: str | None = SchemaField(description="Author name", default=None)
    author_username: str | None = SchemaField(
        description="Author username", default=None
    )
    bookmark_count: int = SchemaField(description="Bookmark count")
    like_count: int = SchemaField(description="Like count")
    quote_count: int = SchemaField(description="Quote count")
    reply_count: int = SchemaField(description="Reply count")
    retweet_count: int = SchemaField(description="Repost count")
    view_count: int = SchemaField(description="View count")
    conversation_id: str | None = SchemaField(
        description="Conversation root post ID", default=None
    )
    in_reply_to_id: str | None = SchemaField(
        description="Parent post ID when this post is a reply", default=None
    )
    is_reply: bool | None = SchemaField(
        description="Whether this post is a reply", default=None
    )
    is_quote_status: bool | None = SchemaField(
        description="Whether this post quotes another post", default=None
    )


class XquikSearchTweetsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = xquik.credentials_field(
            description="The Xquik integration requires an API key."
        )
        query: str = SchemaField(
            description="Words, hashtags, usernames, a post ID, or an X post URL"
        )
        limit: int = SchemaField(
            description="Maximum posts to return",
            default=20,
            ge=1,
            le=100,
            advanced=True,
        )
        sort_order: XquikSortOrder = SchemaField(
            description="Sort by newest posts or engagement",
            default=XquikSortOrder.LATEST,
            advanced=True,
        )
        language: str | None = SchemaField(
            description="Language code such as en or tr",
            default=None,
            advanced=True,
        )
        from_user: str | None = SchemaField(
            description="Only return posts from this username",
            default=None,
            advanced=True,
        )
        since_date: date | None = SchemaField(
            description="Earliest post date",
            default=None,
            advanced=True,
        )
        until_date: date | None = SchemaField(
            description="Latest post date",
            default=None,
            advanced=True,
        )
        cursor: str | None = SchemaField(
            description="Cursor returned by the previous search",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        tweets: list[XquikTweetResult] = SchemaField(description="Matching X posts")
        tweet: XquikTweetResult = SchemaField(description="One matching X post")
        next_cursor: str = SchemaField(description="Cursor for the next page")
        has_next_page: bool = SchemaField(
            description="Whether another result page is available"
        )

    def __init__(self):
        super().__init__(
            id="03373169-7622-4871-8c20-85d893c333e9",
            description=(
                "Searches public X posts through Xquik without requiring an X "
                "developer account"
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": xquik.get_test_credentials().model_dump(),
                "query": "AutoGPT",
            },
            test_credentials=xquik.get_test_credentials(),
            test_output=[
                ("tweets", lambda value: len(value) == 1),
                ("tweet", lambda value: value.id == "1991000000000000000"),
                ("next_cursor", "next-page"),
                ("has_next_page", True),
            ],
            test_mock={"_search": lambda *args, **kwargs: _test_page()},
        )

    async def _search(
        self, credentials: APIKeyCredentials, **kwargs: Unpack[TweetSearchParams]
    ) -> TweetSearchResponse:
        async with AsyncXTwitterScraper(
            api_key=credentials.api_key.get_secret_value()
        ) as client:
            return await client.x.tweets.search(**kwargs)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        search_args: TweetSearchParams = {
            "q": input_data.query,
            "limit": input_data.limit,
            "query_type": input_data.sort_order.value,
        }
        if input_data.language is not None:
            search_args["language"] = input_data.language
        if input_data.from_user is not None:
            search_args["from_user"] = input_data.from_user
        if input_data.since_date is not None:
            search_args["since_date"] = input_data.since_date
        if input_data.until_date is not None:
            search_args["until_date"] = input_data.until_date
        if input_data.cursor is not None:
            search_args["cursor"] = input_data.cursor

        page = await self._search(credentials, **search_args)

        tweets = [_to_result(tweet) for tweet in page.tweets]
        yield "tweets", tweets
        for tweet in tweets:
            yield "tweet", tweet
        if page.next_cursor:
            yield "next_cursor", page.next_cursor
        yield "has_next_page", page.has_next_page


def _to_result(tweet: SearchTweet) -> XquikTweetResult:
    author = tweet.author
    return XquikTweetResult(
        id=tweet.id,
        text=tweet.text,
        url=tweet.url,
        created_at=tweet.created_at,
        language=tweet.lang,
        author_id=author.id if author else None,
        author_name=author.name if author else None,
        author_username=author.username if author else None,
        bookmark_count=tweet.bookmark_count,
        like_count=tweet.like_count,
        quote_count=tweet.quote_count,
        reply_count=tweet.reply_count,
        retweet_count=tweet.retweet_count,
        view_count=tweet.view_count,
        conversation_id=tweet.conversation_id,
        in_reply_to_id=tweet.in_reply_to_id,
        is_reply=tweet.is_reply,
        is_quote_status=tweet.is_quote_status,
    )


def _test_page() -> PaginatedTweets:
    tweet = SearchTweet.model_validate(
        {
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
            "author": {
                "id": "42",
                "name": "Auto-GPT",
                "username": "Auto_GPT",
            },
        }
    )
    return PaginatedTweets(
        tweets=[tweet],
        has_next_page=True,
        next_cursor="next-page",
    )
