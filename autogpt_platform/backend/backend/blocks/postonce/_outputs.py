from collections.abc import Iterator
from typing import Any

from backend.sdk import BlockSchemaOutput, SchemaField


class PostOncePostOutput(BlockSchemaOutput):
    post_id: str = SchemaField(description="PostOnce post ID")
    status: str = SchemaField(
        description="queued, scheduled, processing, partial, published, failed "
        "or cancelled"
    )
    targets: list[dict] = SchemaField(
        description="One entry per account: account_id, platform, username, "
        "status, error, platform_post_id, platform_post_url, scheduled_time"
    )
    post_urls: list[str] = SchemaField(
        description="Links to the published posts; empty until a target publishes"
    )
    post: dict = SchemaField(description="The full post object PostOnce returned")


def post_outputs(post: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    targets = list(post.get("targets") or [])
    yield "post_id", str(post.get("id") or "")
    yield "status", str(post.get("status") or "")
    yield "targets", targets
    yield "post_urls", [
        str(t["platform_post_url"]) for t in targets if t.get("platform_post_url")
    ]
    yield "post", post


TEST_POST_ID = "6f1c2b9e-4d3a-4f8e-9b7c-1a2d3e4f5a6b"


def sample_post(status: str, target_status: str, url: str | None = None) -> dict:
    return {
        "id": TEST_POST_ID,
        "content": "Shipping the public API today.",
        "status": status,
        "publish_at": "2030-04-01T14:00:00.000Z",
        "targets": [
            {
                "target_post_id": "t_1",
                "account_id": "acc_1",
                "platform": "linkedin",
                "username": "postonce",
                "status": target_status,
                "error": None,
                "platform_post_id": "urn:li:share:1" if url else None,
                "platform_post_url": url,
                "scheduled_time": "2030-04-01T14:00:00.000Z",
            }
        ],
    }
