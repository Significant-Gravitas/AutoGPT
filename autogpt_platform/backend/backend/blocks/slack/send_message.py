from backend.blocks.slack._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, SlackCredentials, SlackCredentialsField, SlackCredentialsInput, SlackModel
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from slack_bolt import App
from slack_sdk.errors import SlackApiError

class SlackPostMessageBlock(Block):
    """
    Posts a message to a Slack channel
    """

    class Input(BlockSchema):
        credentials: SlackCredentialsInput = SlackCredentialsField(
            ["chat:write"]
        )

        type: SlackModel = SchemaField(
            title ="type",
            default=SlackModel.SLACK_USER,
            description=(
                "Select how you want to interact with Slack\n\n"
                "User: Interact with Slack using your personal Slack account\n\n"
                "Bot: Interact with Slack as a bot [AutoGPT_Bot] (requires installation in your workspace)"
            ),
            advanced=False,
        )

        channel: str = SchemaField(
            description="Channel ID to post message to (ex: C0XXXXXX)",
            placeholder="Enter channel ID"
        )

        text: str = SchemaField(
            description="Message text to post",
            placeholder="Enter message text"
        )

    class Output(BlockSchema):
        success : bool = SchemaField(description="True if the message was successfully posted to Slack")
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="f8822c4b-b640-11ef-9cc1-a309988b4d92",
            description="Posts a message to a Slack channel",
            categories={BlockCategory.SOCIAL},
            input_schema=SlackPostMessageBlock.Input,
            output_schema=SlackPostMessageBlock.Output,
            test_input={
                "channel": "C0XXXXXX",
                "text": "Test message",
                "type": "user",
                "credentials": TEST_CREDENTIALS_INPUT
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
               ( "success", True)
            ],
            test_mock={
                "post_message": lambda *args, **kwargs: True
            },
        )

    @staticmethod
    def post_message(
        credentials: SlackCredentials,
        channel: str,
        text: str,
    ):
        try:
            app = App(token=credentials.access_token.get_secret_value())
            response = app.client.chat_postMessage(
                channel=channel,
                text=text
            )

            if response["ok"]:
                return True

            raise Exception(response["error"])

        except SlackApiError:
            raise

    def run(
        self,
        input_data: Input,
        *,
        credentials: SlackCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            success = self.post_message(
                credentials,
                input_data.channel,
                input_data.text
            )

            yield "success" , success

        except Exception as e:
            yield "error", str(e)
