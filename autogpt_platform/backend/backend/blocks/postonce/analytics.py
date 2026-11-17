from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaOutput,
    SchemaField,
)

from ._api import PostOnceClient
from ._config import postonce
from ._outputs import TEST_POST_ID
from .posts import PostOncePostIdInput

METRICS = ("views", "likes", "comments", "shares", "saves")


class PostOnceGetPostAnalyticsBlock(Block):
    class Input(PostOncePostIdInput):
        pass

    class Output(BlockSchemaOutput):
        targets: list[dict] = SchemaField(
            description="One entry per published copy on TikTok, Instagram, "
            "YouTube or Facebook: platform, url, views, likes, comments, shares, "
            "saves, engagement_rate, metrics_status, last_updated"
        )
        views: int = SchemaField(description="Views across all copies")
        likes: int = SchemaField(description="Likes across all copies")
        comments: int = SchemaField(description="Comments across all copies")
        shares: int = SchemaField(description="Shares across all copies")
        saves: int = SchemaField(description="Saves across all copies")

    def __init__(self):
        super().__init__(
            id="8bcd9e13-fca0-4567-965e-0c03bd0bc7a5",
            description="Get views, likes, comments, shares and saves for a "
            "published PostOnce post, per platform and in total.",
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
                ("targets", lambda t: len(t) == 2),
                ("views", 1500),
                ("likes", 120),
                ("comments", 9),
                ("shares", 4),
                ("saves", 0),
            ],
            test_mock={"_analytics": lambda *args, **kwargs: SAMPLE_ANALYTICS},
        )

    @staticmethod
    async def _analytics(
        credentials: APIKeyCredentials, post_id: str
    ) -> dict[str, Any]:
        return await PostOnceClient(credentials).get_post_analytics(post_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self._analytics(credentials, input_data.post_id)
        targets = list(data.get("targets") or [])
        yield "targets", targets
        for metric in METRICS:
            # A platform that doesn't report a metric returns null for it.
            yield metric, sum(int(t.get(metric) or 0) for t in targets)


SAMPLE_ANALYTICS = {
    "id": TEST_POST_ID,
    "targets": [
        {
            "account_id": "acc_1",
            "platform": "tiktok",
            "metrics_status": "ok",
            "views": 1000,
            "likes": 100,
            "comments": 5,
            "shares": 4,
            "saves": None,
        },
        {
            "account_id": "acc_2",
            "platform": "instagram",
            "metrics_status": "ok",
            "views": 500,
            "likes": 20,
            "comments": 4,
            "shares": None,
            "saves": None,
        },
    ],
}
