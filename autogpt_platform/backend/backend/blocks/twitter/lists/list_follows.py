# from typing import cast
import tweepy

from backend.blocks.twitter._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    TwitterCredentials,
    TwitterCredentialsField,
    TwitterCredentialsInput,
)

# from backend.blocks.twitter._builders import UserExpansionsBuilder
# from backend.blocks.twitter._types import TweetFields, TweetUserFields, UserExpansionInputs, UserExpansions
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

# from tweepy.client import Response


class TwitterUnfollowListBlock(Block):
    """
    Unfollows a Twitter list for the authenticated user
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["follows.write", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to unfollow",
            placeholder="Enter list ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the unfollow was successful")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="1f43310a-a62f-11ef-8276-2b06a1bbae1a",
            description="This block unfollows a specified Twitter list for the authenticated user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterUnfollowListBlock.Input,
            output_schema=TwitterUnfollowListBlock.Output,
            test_input={"list_id": "123456789", "credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"unfollow_list": lambda *args, **kwargs: True},
        )

    @staticmethod
    def unfollow_list(credentials: TwitterCredentials, list_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.unfollow_list(list_id=list_id, user_auth=False)

            return True

        except tweepy.TweepyException:
            raise

    def run(
        self,
        input_data: Input,
        *,
        credentials: TwitterCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            success = self.unfollow_list(credentials, input_data.list_id)
            yield "success", success
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterFollowListBlock(Block):
    """
    Follows a Twitter list for the authenticated user
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "list.write", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to follow",
            placeholder="Enter list ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the follow was successful")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="03d8acf6-a62f-11ef-b17f-b72b04a09e79",
            description="This block follows a specified Twitter list for the authenticated user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterFollowListBlock.Input,
            output_schema=TwitterFollowListBlock.Output,
            test_input={"list_id": "123456789", "credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"follow_list": lambda *args, **kwargs: True},
        )

    @staticmethod
    def follow_list(credentials: TwitterCredentials, list_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.follow_list(list_id=list_id, user_auth=False)

            return True

        except tweepy.TweepyException:
            raise

    def run(
        self,
        input_data: Input,
        *,
        credentials: TwitterCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            success = self.follow_list(credentials, input_data.list_id)
            yield "success", success
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


# Enterprise Level [Need to do Manual testing], There is a high possibility that we might get error in this
# Needs Type Input in this

# class TwitterListGetFollowersBlock(Block):
#     """
#     Gets followers of a specified Twitter list
#     """

#     class Input(UserExpansionInputs):
#         credentials: TwitterCredentialsInput = TwitterCredentialsField(
#             ["tweet.read","users.read", "list.read", "offline.access"]
#         )

#         list_id: str = SchemaField(
#             description="The ID of the List to get followers for",
#             placeholder="Enter list ID",
#             required=True
#         )

#         max_results: int = SchemaField(
#             description="Max number of results per page (1-100)",
#             placeholder="Enter max results",
#             default=10,
#             advanced=True,
#         )

#         pagination_token: str = SchemaField(
#             description="Token for pagination",
#             placeholder="Enter pagination token",
#             default="",
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         user_ids: list[str] = SchemaField(description="List of user IDs of followers")
#         usernames: list[str] = SchemaField(description="List of usernames of followers")
#         next_token: str = SchemaField(description="Token for next page of results")
#         data: list[dict] = SchemaField(description="Complete follower data")
#         included: dict = SchemaField(description="Additional data requested via expansions")
#         meta: dict = SchemaField(description="Metadata about the response")
#         error: str = SchemaField(description="Error message if the request failed")

#     def __init__(self):
#         super().__init__(
#             id="16b289b4-a62f-11ef-95d4-bb29b849eb99",
#             description="This block retrieves followers of a specified Twitter list.",
#             categories={BlockCategory.SOCIAL},
#             input_schema=TwitterListGetFollowersBlock.Input,
#             output_schema=TwitterListGetFollowersBlock.Output,
#             test_input={
#                 "list_id": "123456789",
#                 "max_results": 10,
#                 "pagination_token": None,
#                 "credentials": TEST_CREDENTIALS_INPUT,
#                 "expansions": [],
#                 "tweet_fields": [],
#                 "user_fields": []
#             },
#             test_credentials=TEST_CREDENTIALS,
#             test_output=[
#                 ("user_ids", ["2244994945"]),
#                 ("usernames", ["testuser"]),
#                 ("next_token", None),
#                 ("data", {"followers": [{"id": "2244994945", "username": "testuser"}]}),
#                 ("included", {}),
#                 ("meta", {}),
#                 ("error", "")
#             ],
#             test_mock={
#                 "get_list_followers": lambda *args, **kwargs: ({
#                     "followers": [{"id": "2244994945", "username": "testuser"}]
#                 }, {}, {}, ["2244994945"], ["testuser"], None)
#             }
#         )

#     @staticmethod
#     def get_list_followers(
#         credentials: TwitterCredentials,
#         list_id: str,
#         max_results: int,
#         pagination_token: str,
#         expansions: list[UserExpansions],
#         tweet_fields: list[TweetFields],
#         user_fields: list[TweetUserFields]
#     ):
#         try:
#             client = tweepy.Client(
#                 bearer_token=credentials.access_token.get_secret_value(),
#             )

#             params = {
#                 "id": list_id,
#                 "max_results": max_results,
#                 "pagination_token": None if pagination_token == "" else pagination_token,
#                 "user_auth": False
#             }

#             params = (UserExpansionsBuilder(params)
#                     .add_expansions(expansions)
#                     .add_tweet_fields(tweet_fields)
#                     .add_user_fields(user_fields)
#                     .build())

#             response = cast(
#                 Response,
#                 client.get_list_followers(**params)
#             )

#             meta = {}
#             user_ids = []
#             usernames = []
#             next_token = None

#             if response.meta:
#                 meta = response.meta
#                 next_token = meta.get("next_token")

#             included = IncludesSerializer.serialize(response.includes)
#             data = ResponseDataSerializer.serialize_list(response.data)

#             if response.data:
#                 user_ids = [str(item.id) for item in response.data]
#                 usernames = [item.username for item in response.data]

#                 return data, included, meta, user_ids, usernames, next_token

#             raise Exception("No followers found")

#         except tweepy.TweepyException:
#             raise

#     def run(
#         self,
#         input_data: Input,
#         *,
#         credentials: TwitterCredentials,
#         **kwargs,
#     ) -> BlockOutput:
#         try:
#             followers_data, included, meta, user_ids, usernames, next_token = self.get_list_followers(
#                 credentials,
#                 input_data.list_id,
#                 input_data.max_results,
#                 input_data.pagination_token,
#                 input_data.expansions,
#                 input_data.tweet_fields,
#                 input_data.user_fields
#             )

#             if user_ids:
#                 yield "user_ids", user_ids
#             if usernames:
#                 yield "usernames", usernames
#             if next_token:
#                 yield "next_token", next_token
#             if followers_data:
#                 yield "data", followers_data
#             if included:
#                 yield "included", included
#             if meta:
#                 yield "meta", meta

#         except Exception as e:
#             yield "error", handle_tweepy_exception(e)

# class TwitterGetFollowedListsBlock(Block):
#     """
#     Gets lists followed by a specified Twitter user
#     """

#     class Input(UserExpansionInputs):
#         credentials: TwitterCredentialsInput = TwitterCredentialsField(
#             ["follows.read", "users.read", "list.read", "offline.access"]
#         )

#         user_id: str = SchemaField(
#             description="The user ID whose followed Lists to retrieve",
#             placeholder="Enter user ID",
#             required=True
#         )

#         max_results: int = SchemaField(
#             description="Max number of results per page (1-100)",
#             placeholder="Enter max results",
#             default=10,
#             advanced=True,
#         )

#         pagination_token: str = SchemaField(
#             description="Token for pagination",
#             placeholder="Enter pagination token",
#             default="",
#             advanced=True,
#         )

#     class Output(BlockSchema):
#         list_ids: list[str] = SchemaField(description="List of list IDs")
#         list_names: list[str] = SchemaField(description="List of list names")
#         data: list[dict] = SchemaField(description="Complete list data")
#         includes: dict = SchemaField(description="Additional data requested via expansions")
#         meta: dict = SchemaField(description="Metadata about the response")
#         next_token: str = SchemaField(description="Token for next page of results")
#         error: str = SchemaField(description="Error message if the request failed")

#     def __init__(self):
#         super().__init__(
#             id="0e18bbfc-a62f-11ef-94fa-1f1e174b809e",
#             description="This block retrieves all Lists a specified user follows.",
#             categories={BlockCategory.SOCIAL},
#             input_schema=TwitterGetFollowedListsBlock.Input,
#             output_schema=TwitterGetFollowedListsBlock.Output,
#             test_input={
#                 "user_id": "123456789",
#                 "max_results": 10,
#                 "pagination_token": None,
#                 "credentials": TEST_CREDENTIALS_INPUT,
#                 "expansions": [],
#                 "tweet_fields": [],
#                 "user_fields": []
#             },
#             test_credentials=TEST_CREDENTIALS,
#             test_output=[
#                 ("list_ids", ["12345"]),
#                 ("list_names", ["Test List"]),
#                 ("data", {"followed_lists": [{"id": "12345", "name": "Test List"}]}),
#                 ("includes", {}),
#                 ("meta", {}),
#                 ("next_token", None),
#                 ("error", "")
#             ],
#             test_mock={
#                 "get_followed_lists": lambda *args, **kwargs: ({
#                     "followed_lists": [{"id": "12345", "name": "Test List"}]
#                 }, {}, {}, ["12345"], ["Test List"], None)
#             }
#         )

#     @staticmethod
#     def get_followed_lists(
#         credentials: TwitterCredentials,
#         user_id: str,
#         max_results: int,
#         pagination_token: str,
#         expansions: list[UserExpansions],
#         tweet_fields: list[TweetFields],
#         user_fields: list[TweetUserFields]
#     ):
#         try:
#             client = tweepy.Client(
#                 bearer_token=credentials.access_token.get_secret_value(),
#             )

#             params = {
#                 "id": user_id,
#                 "max_results": max_results,
#                 "pagination_token": None if pagination_token == "" else pagination_token,
#                 "user_auth": False
#             }

#             params = (UserExpansionsBuilder(params)
#                     .add_expansions(expansions)
#                     .add_tweet_fields(tweet_fields)
#                     .add_user_fields(user_fields)
#                     .build())

#             response = cast(
#                 Response,
#                 client.get_followed_lists(**params)
#             )

#             meta = {}
#             list_ids = []
#             list_names = []
#             next_token = None

#             if response.meta:
#                 meta = response.meta
#                 next_token = meta.get("next_token")

#             included = IncludesSerializer.serialize(response.includes)
#             data = ResponseDataSerializer.serialize_list(response.data)

#             if response.data:
#                 list_ids = [str(item.id) for item in response.data]
#                 list_names = [item.name for item in response.data]

#                 return data, included, meta, list_ids, list_names, next_token

#             raise Exception("No followed lists found")

#         except tweepy.TweepyException:
#             raise

#     def run(
#         self,
#         input_data: Input,
#         *,
#         credentials: TwitterCredentials,
#         **kwargs,
#     ) -> BlockOutput:
#         try:
#             lists_data, included, meta, list_ids, list_names, next_token = self.get_followed_lists(
#                 credentials,
#                 input_data.user_id,
#                 input_data.max_results,
#                 input_data.pagination_token,
#                 input_data.expansions,
#                 input_data.tweet_fields,
#                 input_data.user_fields
#             )

#             if list_ids:
#                 yield "list_ids", list_ids
#             if list_names:
#                 yield "list_names", list_names
#             if next_token:
#                 yield "next_token", next_token
#             if lists_data:
#                 yield "data", lists_data
#             if included:
#                 yield "includes", included
#             if meta:
#                 yield "meta", meta

#         except Exception as e:
#             yield "error", handle_tweepy_exception(e)
