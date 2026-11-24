from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import json_object, status_request
from ._base import MrScraperBlockBase
from ._config import mrscraper, mrscraper_credentials_field


class MrScraperStatusBlock(MrScraperBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = mrscraper_credentials_field()

    class Output(BlockSchemaOutput):
        response: dict[str, Any] = SchemaField(
            description="Raw account status API response, with message and data."
        )
        account: dict[str, Any] = SchemaField(
            description="Account details from the response's data field, including "
            "subscriptionItem and scraperAllowance."
        )

    def __init__(self):
        super().__init__(
            id="cc3b76ba-66c6-4f89-8dc0-54206e9e7007",
            description="Return the raw account status API response without "
            "summarizing, redacting body fields, or adding calculated fields.",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.READ,
            test_input={"credentials": mrscraper.get_test_credentials().model_dump()},
            test_credentials=mrscraper.get_test_credentials(),
            test_output=[
                ("response", lambda r: r["data"]["tokenLimit"] == 100000),
                ("account", lambda a: a["scraperAllowance"]["remaining"] == 97500),
            ],
            test_mock={
                "_send": lambda *args, **kwargs: {
                    "message": "Example response",
                    "data": {
                        "id": "ACCOUNT_ID",
                        "tokenLimit": 100000,
                        "tokenUsage": 2500,
                        "scraperAllowance": {"remaining": 97500},
                    },
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        body = json_object(await self._send(credentials, status_request()))
        account = body.get("data")
        yield "response", body
        yield "account", account if isinstance(account, dict) else {}
