# Todo : Add new Type support

# from typing import cast
# import tweepy
# from tweepy.client import Response

# from backend.blocks.twitter._serializer import IncludesSerializer, ResponseDataSerializer
# from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
# from backend.data.model import SchemaField
# from backend.blocks.twitter._builders import DMExpansionsBuilder
# from backend.blocks.twitter._types import DMEventExpansion, DMEventExpansionInputs, DMEventType, DMMediaField, DMTweetField, TweetUserFields
# from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
# from backend.blocks.twitter._auth import (
#     TEST_CREDENTIALS,
#     TEST_CREDENTIALS_INPUT,
#     TwitterCredentials,
#     TwitterCredentialsField,
#     TwitterCredentialsInput,
# )

# Require Pro or Enterprise plan [Manual Testing Required]
# class TwitterGetDMEventsBlock(Block):
#     """
#     Gets a list of Direct Message events for the authenticated user
#     """

#     class Input(DMEventExpansionInputs):
#         credentials: TwitterCredentialsInput = TwitterCredentialsField(
#             ["dm.read", "offline.access", "user.read", "tweet.read"]
#         )

#         dm_conversation_id: str = SchemaField(
#             description="The ID of the Direct Message conversation",
#             placeholder="Enter conversation ID",
#             required=True
#         )

#         max_results: int = SchemaField(
#             description="Maximum number of results to return (1-100)",
#             placeholder="Enter max results",
#             advanced=True,
#             default=10,
#         )

#         pagination_token: str = SchemaField(
#             description="Token for pagination",
#             placeholder="Enter pagination token",
#             advanced=True,
#             default=""
#         )

#     class Output(BlockSchema):
#         # Common outputs
#         event_ids: list[str] = SchemaField(description="DM Event IDs")
#         event_texts: list[str] = SchemaField(description="DM Event text contents")
#         event_types: list[str] = SchemaField(description="Types of DM events")
#         next_token: str = SchemaField(description="Token for next page of results")

#         # Complete outputs
#         data: list[dict] = SchemaField(description="Complete DM events data")
#         included: dict = SchemaField(description="Additional data requested via expansions")
#         meta: dict = SchemaField(description="Metadata about the response")
#         error: str = SchemaField(description="Error message if request failed")

#     def __init__(self):
#         super().__init__(
#             id="dc37a6d4-a62e-11ef-a3a5-03061375737b",
#             description="This block retrieves Direct Message events for the authenticated user.",
#             categories={BlockCategory.SOCIAL},
#             input_schema=TwitterGetDMEventsBlock.Input,
#             output_schema=TwitterGetDMEventsBlock.Output,
#             test_input={
#                 "dm_conversation_id": "1234567890",
#                 "max_results": 10,
#                 "credentials": TEST_CREDENTIALS_INPUT,
#                 "expansions": [],
#                 "event_types": [],
#                 "media_fields": [],
#                 "tweet_fields": [],
#                 "user_fields": []
#             },
#             test_credentials=TEST_CREDENTIALS,
#             test_output=[
#                 ("event_ids", ["1346889436626259968"]),
#                 ("event_texts", ["Hello just you..."]),
#                 ("event_types", ["MessageCreate"]),
#                 ("next_token", None),
#                 ("data", [{"id": "1346889436626259968", "text": "Hello just you...", "event_type": "MessageCreate"}]),
#                 ("included", {}),
#                 ("meta", {}),
#                 ("error", "")
#             ],
#             test_mock={
#                 "get_dm_events": lambda *args, **kwargs: (
#                     [{"id": "1346889436626259968", "text": "Hello just you...", "event_type": "MessageCreate"}],
#                     {},
#                     {},
#                     ["1346889436626259968"],
#                     ["Hello just you..."],
#                     ["MessageCreate"],
#                     None
#                 )
#             }
#         )

#     @staticmethod
#     def get_dm_events(
#         credentials: TwitterCredentials,
#         dm_conversation_id: str,
#         max_results: int,
#         pagination_token: str,
#         expansions: list[DMEventExpansion],
#         event_types: list[DMEventType],
#         media_fields: list[DMMediaField],
#         tweet_fields: list[DMTweetField],
#         user_fields: list[TweetUserFields]
#     ):
#         try:
#             client = tweepy.Client(
#                 bearer_token=credentials.access_token.get_secret_value()
#             )

#             params = {
#                 "dm_conversation_id": dm_conversation_id,
#                 "max_results": max_results,
#                 "pagination_token": None if pagination_token == "" else pagination_token,
#                 "user_auth": False
#             }

#             params = (DMExpansionsBuilder(params)
#                      .add_expansions(expansions)
#                      .add_event_types(event_types)
#                      .add_media_fields(media_fields)
#                      .add_tweet_fields(tweet_fields)
#                      .add_user_fields(user_fields)
#                      .build())

#             response = cast(Response, client.get_direct_message_events(**params))

#             meta = {}
#             event_ids = []
#             event_texts = []
#             event_types = []
#             next_token = None

#             if response.meta:
#                 meta = response.meta
#                 next_token = meta.get("next_token")

#             included = IncludesSerializer.serialize(response.includes)
#             data = ResponseDataSerializer.serialize_list(response.data)

#             if response.data:
#                 event_ids = [str(item.id) for item in response.data]
#                 event_texts = [item.text if hasattr(item, "text") else None for item in response.data]
#                 event_types = [item.event_type for item in response.data]

#                 return data, included, meta, event_ids, event_texts, event_types, next_token

#             raise Exception("No DM events found")

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
#             event_data, included, meta, event_ids, event_texts, event_types, next_token = self.get_dm_events(
#                 credentials,
#                 input_data.dm_conversation_id,
#                 input_data.max_results,
#                 input_data.pagination_token,
#                 input_data.expansions,
#                 input_data.event_types,
#                 input_data.media_fields,
#                 input_data.tweet_fields,
#                 input_data.user_fields
#             )

#             if event_ids:
#                 yield "event_ids", event_ids
#             if event_texts:
#                 yield "event_texts", event_texts
#             if event_types:
#                 yield "event_types", event_types
#             if next_token:
#                 yield "next_token", next_token
#             if event_data:
#                 yield "data", event_data
#             if included:
#                 yield "included", included
#             if meta:
#                 yield "meta", meta

#         except Exception as e:
#             yield "error", handle_tweepy_exception(e)
