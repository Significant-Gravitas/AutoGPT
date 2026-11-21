from enum import Enum
from typing import Any, Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import PostOnceClient
from ._config import CREDENTIALS_DESCRIPTION, postonce


class PostOncePlatform(str, Enum):
    BLUESKY = "bluesky"
    FACEBOOK = "facebook"
    FARCASTER = "farcaster"
    GOOGLE_BUSINESS = "google_business"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    NOSTR = "nostr"
    PINTEREST = "pinterest"
    REDDIT = "reddit"
    SNAPCHAT = "snapchat"
    TELEGRAM = "telegram"
    THREADS = "threads"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    VIMEO = "vimeo"
    YOUTUBE = "youtube"


class PostOnceListAccountsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = postonce.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        platform: Optional[PostOncePlatform] = SchemaField(
            description="Only list accounts on this platform", default=None
        )
        active_only: bool = SchemaField(
            description="Leave out accounts that need reconnecting",
            default=True,
        )

    class Output(BlockSchemaOutput):
        accounts: list[dict] = SchemaField(
            description="Connected accounts: id, platform, username, status, "
            "capabilities, media_requirements"
        )
        account_ids: list[str] = SchemaField(
            description="IDs of the listed accounts, ready for Create Post"
        )

    def __init__(self):
        super().__init__(
            id="a911ef09-bae8-40cd-b9ab-024ea7e20901",
            description="List the social accounts connected to PostOnce, with the "
            "IDs Create Post needs.",
            categories={BlockCategory.SOCIAL},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": postonce.get_test_credentials().model_dump(),
                "platform": "linkedin",
            },
            test_credentials=postonce.get_test_credentials(),
            test_output=[
                ("accounts", lambda a: [x["id"] for x in a] == ["acc_1"]),
                ("account_ids", ["acc_1"]),
            ],
            test_mock={"_list": lambda *args, **kwargs: SAMPLE_ACCOUNTS},
        )

    @staticmethod
    async def _list(credentials: APIKeyCredentials) -> list[dict[str, Any]]:
        return await PostOnceClient(credentials).list_accounts()

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        accounts = [
            a
            for a in await self._list(credentials)
            if (not input_data.active_only or a.get("status") == "active")
            and (
                input_data.platform is None
                or a.get("platform") == input_data.platform.value
            )
        ]
        yield "accounts", accounts
        yield "account_ids", [str(a.get("id")) for a in accounts]


SAMPLE_ACCOUNTS = [
    {"id": "acc_1", "platform": "linkedin", "username": "a", "status": "active"},
    {"id": "acc_2", "platform": "linkedin", "username": "b", "status": "auth_required"},
    {"id": "acc_3", "platform": "tiktok", "username": "c", "status": "active"},
]
