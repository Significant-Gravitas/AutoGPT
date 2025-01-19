import requests
import json
from datetime import datetime
from pydantic import BaseModel, ConfigDict

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField
from backend.util.mock import MockObject

class PostTwitterTweetBlock(Block):
    api_url: str = "https://api.twitter.com/2"

    class Input(BlockSchema):
        post_content: str = SchemaField(
            description="Twitter post",
            default=f"Hello, Twitter! This tweet was posted using OAuth 2.0 User Context. 🚀 {datetime.now().isoformat()}")
        access_token: str = SchemaField(
            description="Twitter credentials", 
        )

    class Output(BlockSchema):
        post_id: str = SchemaField(description="Posted ID")

    def __init__(self):
        super().__init__(
            id="dc4bd1eb-a4f3-4ab8-a3d8-b80e1a1d959e",
            description="This block posts a tweet to Twitter.",
            categories={BlockCategory.SOCIAL},
            input_schema=PostTwitterTweetBlock.Input,
            output_schema=PostTwitterTweetBlock.Output,
            test_input={
                "access_token": "access_token",
                "post_content": "post_content"
            },
            test_output=[("post_id", "post_id")],
            test_mock={"post_tweet": lambda access_token, post_content: "post_id"},
        )

    @staticmethod
    def post_tweet(access_token: str, post_content: str) -> str:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # Data for the tweet
        body = {
            "text": post_content[:255] if len(post_content) > 255 else post_content,
        }

        response = requests.post(PostTwitterTweetBlock.api_url + "/tweets", headers=headers, data=json.dumps(body))

        if response.status_code != 201:
            raise ValueError("Failed to post a tweet because of an error. Error: " + response.text)

        return response.json().get("data").get("id")

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "post_id", self.post_tweet(input_data.access_token, input_data.post_content)
