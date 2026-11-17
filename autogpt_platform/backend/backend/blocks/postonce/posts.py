from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import PostOnceClient
from ._config import CREDENTIALS_DESCRIPTION, postonce
from ._outputs import TEST_POST_ID, PostOncePostOutput, post_outputs, sample_post

PUBLISHED_URL = "https://www.linkedin.com/feed/update/urn:li:share:1"


class PostOncePostIdInput(BlockSchemaInput):
    credentials: CredentialsMetaInput = postonce.credentials_field(
        description=CREDENTIALS_DESCRIPTION
    )
    post_id: str = SchemaField(
        description="PostOnce post ID, from the post_id output of Create Post"
    )


class PostOnceGetPostBlock(Block):
    class Input(PostOncePostIdInput):
        pass

    class Output(PostOncePostOutput):
        pass

    def __init__(self):
        super().__init__(
            id="7eec96db-2918-4c77-a278-7f4bb77a915c",
            description="Get a PostOnce post's status and, for each account, its "
            "delivery state, error and published link.",
            categories={BlockCategory.SOCIAL},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": postonce.get_test_credentials().model_dump(),
                "post_id": TEST_POST_ID,
            },
            test_credentials=postonce.get_test_credentials(),
            test_output=[
                ("post_id", TEST_POST_ID),
                ("status", "published"),
                ("targets", lambda t: t[0]["status"] == "published"),
                ("post_urls", [PUBLISHED_URL]),
                ("post", lambda p: p["id"] == TEST_POST_ID),
            ],
            test_mock={
                "_get": lambda *args, **kwargs: sample_post(
                    "published", "published", PUBLISHED_URL
                )
            },
        )

    @staticmethod
    async def _get(credentials: APIKeyCredentials, post_id: str) -> dict[str, Any]:
        return await PostOnceClient(credentials).get_post(post_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        post = await self._get(credentials, input_data.post_id)
        for output in post_outputs(post):
            yield output


class PostOnceCancelPostBlock(Block):
    class Input(PostOncePostIdInput):
        pass

    class Output(PostOncePostOutput):
        pass

    def __init__(self):
        super().__init__(
            id="1cdd8359-0477-4b85-aaea-bd02b61c7f3b",
            description="Cancel a scheduled or still-pending PostOnce post. "
            "Copies that already went out stay published.",
            categories={BlockCategory.SOCIAL},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            test_input={
                "credentials": postonce.get_test_credentials().model_dump(),
                "post_id": TEST_POST_ID,
            },
            test_credentials=postonce.get_test_credentials(),
            test_output=[
                ("post_id", TEST_POST_ID),
                ("status", "cancelled"),
                ("targets", lambda t: t[0]["status"] == "cancelled"),
                ("post_urls", []),
                ("post", lambda p: p["status"] == "cancelled"),
            ],
            test_mock={
                "_cancel": lambda *args, **kwargs: sample_post("cancelled", "cancelled")
            },
        )

    @staticmethod
    async def _cancel(credentials: APIKeyCredentials, post_id: str) -> dict[str, Any]:
        return await PostOnceClient(credentials).cancel_post(post_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        post = await self._cancel(credentials, input_data.post_id)
        for output in post_outputs(post):
            yield output
