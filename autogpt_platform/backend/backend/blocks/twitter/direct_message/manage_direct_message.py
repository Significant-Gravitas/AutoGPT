# Todo : Add new Type support

# from typing import cast

# import tweepy
# from tweepy.client import Response

# from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
# from backend.data.model import SchemaField
# from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
# from backend.blocks.twitter._auth import (
#     TEST_CREDENTIALS,
#     TEST_CREDENTIALS_INPUT,
#     TwitterCredentials,
#     TwitterCredentialsField,
#     TwitterCredentialsInput,
# )

# Pro and Enterprise plan [Manual Testing Required]
# class TwitterSendDirectMessageBlock(Block):
#     """
#     Sends a direct message to a Twitter user
#     """

#     class Input(BlockSchema):
#         credentials: TwitterCredentialsInput = TwitterCredentialsField(
#             ["offline.access", "direct_messages.write"]
#         )

#         participant_id: str = SchemaField(
#             description="The User ID of the account to send DM to",
#             placeholder="Enter recipient user ID",
#             default="",
#             advanced=False
#         )

#         dm_conversation_id: str = SchemaField(
#             description="The conversation ID to send message to",
#             placeholder="Enter conversation ID",
#             default="",
#             advanced=False
#         )

#         text: str = SchemaField(
#             description="Text of the Direct Message (up to 10,000 characters)",
#             placeholder="Enter message text",
#             default="",
#             advanced=False
#         )

#         media_id: str = SchemaField(
#             description="Media ID to attach to the message",
#             placeholder="Enter media ID",
#             default=""
#         )

#     class Output(BlockSchema):
#         dm_event_id: str = SchemaField(description="ID of the sent direct message")
#         dm_conversation_id_: str = SchemaField(description="ID of the conversation")
#         error: str = SchemaField(description="Error message if sending failed")

#     def __init__(self):
#         super().__init__(
#             id="f32f2786-a62e-11ef-a93d-a3ef199dde7f",
#             description="This block sends a direct message to a specified Twitter user.",
#             categories={BlockCategory.SOCIAL},
#             input_schema=TwitterSendDirectMessageBlock.Input,
#             output_schema=TwitterSendDirectMessageBlock.Output,
#             test_input={
#                 "participant_id": "783214",
#                 "dm_conversation_id": "",
#                 "text": "Hello from Twitter API",
#                 "media_id": "",
#                 "credentials": TEST_CREDENTIALS_INPUT
#             },
#             test_credentials=TEST_CREDENTIALS,
#             test_output=[
#                 ("dm_event_id", "0987654321"),
#                 ("dm_conversation_id_", "1234567890"),
#                 ("error", "")
#             ],
#             test_mock={
#                 "send_direct_message": lambda *args, **kwargs: (
#                     "0987654321",
#                     "1234567890"
#                 )
#             },
#         )

#     @staticmethod
#     def send_direct_message(
#         credentials: TwitterCredentials,
#         participant_id: str,
#         dm_conversation_id: str,
#         text: str,
#         media_id: str
#     ):
#         try:
#             client = tweepy.Client(
#                 bearer_token=credentials.access_token.get_secret_value()
#             )

#             response = cast(
#                 Response,
#                 client.create_direct_message(
#                     participant_id=None if participant_id == "" else participant_id,
#                     dm_conversation_id=None if dm_conversation_id == "" else dm_conversation_id,
#                     text=None if text == "" else text,
#                     media_id=None if media_id == "" else media_id,
#                     user_auth=False
#                 )
#             )

#             if not response.data:
#                 raise Exception("Failed to send direct message")

#             return response.data["dm_event_id"], response.data["dm_conversation_id"]

#         except tweepy.TweepyException:
#             raise
#         except Exception as e:
#             print(f"Unexpected error: {str(e)}")
#             raise

#     def run(
#         self,
#         input_data: Input,
#         *,
#         credentials: TwitterCredentials,
#         **kwargs,
#     ) -> BlockOutput:
#         try:
#             dm_event_id, dm_conversation_id = self.send_direct_message(
#                 credentials,
#                 input_data.participant_id,
#                 input_data.dm_conversation_id,
#                 input_data.text,
#                 input_data.media_id
#             )
#             yield "dm_event_id", dm_event_id
#             yield "dm_conversation_id", dm_conversation_id

#         except Exception as e:
#             yield "error", handle_tweepy_exception(e)

# class TwitterCreateDMConversationBlock(Block):
#     """
#     Creates a new group direct message conversation on Twitter
#     """

#     class Input(BlockSchema):
#         credentials: TwitterCredentialsInput = TwitterCredentialsField(
#             ["offline.access", "dm.write","dm.read","tweet.read","user.read"]
#         )

#         participant_ids: list[str] = SchemaField(
#             description="Array of User IDs to create conversation with (max 50)",
#             placeholder="Enter participant user IDs",
#             default=[],
#             advanced=False
#         )

#         text: str = SchemaField(
#             description="Text of the Direct Message (up to 10,000 characters)",
#             placeholder="Enter message text",
#             default="",
#             advanced=False
#         )

#         media_id: str = SchemaField(
#             description="Media ID to attach to the message",
#             placeholder="Enter media ID",
#             default="",
#             advanced=False
#         )

#     class Output(BlockSchema):
#         dm_event_id: str = SchemaField(description="ID of the sent direct message")
#         dm_conversation_id: str = SchemaField(description="ID of the conversation")
#         error: str = SchemaField(description="Error message if sending failed")

#     def __init__(self):
#         super().__init__(
#             id="ec11cabc-a62e-11ef-8c0e-3fe37ba2ec92",
#             description="This block creates a new group DM conversation with specified Twitter users.",
#             categories={BlockCategory.SOCIAL},
#             input_schema=TwitterCreateDMConversationBlock.Input,
#             output_schema=TwitterCreateDMConversationBlock.Output,
#             test_input={
#                 "participant_ids": ["783214", "2244994945"],
#                 "text": "Hello from Twitter API",
#                 "media_id": "",
#                 "credentials": TEST_CREDENTIALS_INPUT
#             },
#             test_credentials=TEST_CREDENTIALS,
#             test_output=[
#                 ("dm_event_id", "0987654321"),
#                 ("dm_conversation_id", "1234567890"),
#                 ("error", "")
#             ],
#             test_mock={
#                 "create_dm_conversation": lambda *args, **kwargs: (
#                     "0987654321",
#                     "1234567890"
#                 )
#             },
#         )

#     @staticmethod
#     def create_dm_conversation(
#         credentials: TwitterCredentials,
#         participant_ids: list[str],
#         text: str,
#         media_id: str
#     ):
#         try:
#             client = tweepy.Client(
#                 bearer_token=credentials.access_token.get_secret_value()
#             )

#             response = cast(
#                 Response,
#                 client.create_direct_message_conversation(
#                     participant_ids=participant_ids,
#                     text=None if text == "" else text,
#                     media_id=None if media_id == "" else media_id,
#                     user_auth=False
#                 )
#             )

#             if not response.data:
#                 raise Exception("Failed to create DM conversation")

#             return response.data["dm_event_id"], response.data["dm_conversation_id"]

#         except tweepy.TweepyException:
#             raise
#         except Exception as e:
#             print(f"Unexpected error: {str(e)}")
#             raise

#     def run(
#         self,
#         input_data: Input,
#         *,
#         credentials: TwitterCredentials,
#         **kwargs,
#     ) -> BlockOutput:
#         try:
#             dm_event_id, dm_conversation_id = self.create_dm_conversation(
#                 credentials,
#                 input_data.participant_ids,
#                 input_data.text,
#                 input_data.media_id
#             )
#             yield "dm_event_id", dm_event_id
#             yield "dm_conversation_id", dm_conversation_id

#         except Exception as e:
#             yield "error", handle_tweepy_exception(e)
