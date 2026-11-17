from datetime import datetime, timezone
from typing import Any, Optional

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
from ._outputs import PostOncePostOutput, post_outputs, sample_post


class PostOnceCreatePostBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = postonce.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        account_ids: list[str] = SchemaField(
            description="IDs of the connected accounts to post to, from PostOnce "
            "List Accounts",
            min_length=1,
        )
        content: str = SchemaField(description="Post text", default="")
        media_urls: list[str] = SchemaField(
            description="Public HTTP(S) URLs of images or videos to attach",
            default_factory=list,
        )
        publish_at: Optional[datetime] = SchemaField(
            description="When to publish (UTC if no time zone is given). Leave "
            "empty to publish now.",
            default=None,
        )
        content_overrides: dict[str, str] = SchemaField(
            description="Different text for some accounts, keyed by account ID",
            default_factory=dict,
            advanced=True,
        )
        platform_options: dict[str, dict] = SchemaField(
            description="Platform settings keyed by account ID, e.g. "
            '{"<youtube account>": {"title": "Launch video"}}',
            default_factory=dict,
            advanced=True,
        )
        external_id: str = SchemaField(
            description="Your own reference for this post", default="", advanced=True
        )
        idempotency_key: str = SchemaField(
            description="Reusing a key with the same input returns the original "
            "post instead of posting twice",
            default="",
            advanced=True,
        )

    class Output(PostOncePostOutput):
        pass

    def __init__(self):
        super().__init__(
            id="94d35374-809d-48f6-8549-955ff05093b9",
            description="Publish or schedule one post to several social accounts "
            "with PostOnce, with optional media and per-account text.",
            categories={BlockCategory.SOCIAL},
            input_schema=self.Input,
            output_schema=self.Output,
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
            test_input={
                "credentials": postonce.get_test_credentials().model_dump(),
                "account_ids": ["acc_1"],
                "content": "Shipping the public API today.",
                "publish_at": "2030-04-01T14:00:00Z",
            },
            test_credentials=postonce.get_test_credentials(),
            test_output=[
                ("post_id", sample_post("scheduled", "scheduled")["id"]),
                ("status", "scheduled"),
                ("targets", lambda t: t[0]["account_id"] == "acc_1"),
                ("post_urls", []),
                ("post", lambda p: p["status"] == "scheduled"),
            ],
            test_mock={
                "_create": lambda *args, **kwargs: sample_post("scheduled", "scheduled")
            },
        )

    @staticmethod
    async def _create(
        credentials: APIKeyCredentials, payload: dict[str, Any], idempotency_key: str
    ) -> dict[str, Any]:
        return await PostOnceClient(credentials).create_post(payload, idempotency_key)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        post = await self._create(
            credentials, build_payload(input_data), input_data.idempotency_key
        )
        for output in post_outputs(post):
            yield output


def build_payload(input_data: PostOnceCreatePostBlock.Input) -> dict[str, Any]:
    unknown = (
        set(input_data.content_overrides) | set(input_data.platform_options)
    ) - set(input_data.account_ids)
    if unknown:
        raise ValueError(
            "content_overrides and platform_options name accounts that are not "
            f"in account_ids: {', '.join(sorted(unknown))}"
        )
    if not input_data.content.strip() and not input_data.media_urls:
        raise ValueError("Give the post some content or at least one media URL")

    payload: dict[str, Any] = {
        "content": input_data.content,
        "targets": [_target(input_data, a) for a in input_data.account_ids],
    }
    if input_data.media_urls:
        payload["media"] = [{"url": url} for url in input_data.media_urls]
    if input_data.publish_at:
        payload["publish_at"] = _utc_iso(input_data.publish_at)
    if input_data.external_id:
        payload["external_id"] = input_data.external_id
    return payload


def _target(input_data: PostOnceCreatePostBlock.Input, account_id: str) -> dict:
    target: dict[str, Any] = {"account_id": account_id}
    if account_id in input_data.content_overrides:
        target["content_override"] = input_data.content_overrides[account_id]
    if account_id in input_data.platform_options:
        target["platform_options"] = input_data.platform_options[account_id]
    return target


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
