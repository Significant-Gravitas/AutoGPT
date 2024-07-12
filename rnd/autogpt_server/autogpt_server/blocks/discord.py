import requests
from autogpt_server.data.block import Block, BlockSchema, BlockOutput


class DiscordSendMessage(Block):
    class Input(BlockSchema):
        webhook_url: str
        message: str

    class Output(BlockSchema):
        status: str

    def __init__(self):
        super().__init__(
            id="b3a9c1f2-5d4e-47b3-9c4e-2b6e4d2c4f3e",
            input_schema=DiscordSendMessage.Input,
            output_schema=DiscordSendMessage.Output,
            test_input={
                "webhook_url": "https://discord.com/api/webhooks/your_webhook_url",
                "message": "Hello, Webhook!"
            },
            test_output=("status", "sent"),
        )

    def run(self, input_data: Input) -> BlockOutput:
        response = requests.post(input_data.webhook_url, json={"content": input_data.message})
        if response.status_code == 204:  # Discord webhook returns 204 No Content on success
            yield "status", "sent"
        else:
            yield "status", f"failed with status code {response.status_code}"
