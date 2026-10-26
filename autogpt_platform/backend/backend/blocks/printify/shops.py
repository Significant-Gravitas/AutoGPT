from backend.blocks.printify._api import PrintifyClient, PrintifyShop
from backend.blocks.printify._config import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    PrintifyCredentials,
    PrintifyCredentialsInput,
    printify,
)
from backend.sdk import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    SchemaField,
)

_TEST_SHOP = PrintifyShop(id=8291, title="Sticker Studio", sales_channel="etsy")


class PrintifyListShopsBlock(Block):
    """List the Printify shops on the account and the store each sells through."""

    class Input(BlockSchemaInput):
        credentials: PrintifyCredentialsInput = printify.credentials_field(
            description="Your Printify personal access token"
        )

    class Output(BlockSchemaOutput):
        shops: list[PrintifyShop] = SchemaField(
            description="Every shop on the account", default_factory=list
        )
        shop: PrintifyShop = SchemaField(description="Each shop, one at a time")

    def __init__(self):
        super().__init__(
            id="02d69afa-da23-4de9-ae2a-fe376eb9ce2e",
            description=(
                "List your Printify shops and the store each one publishes to "
                "(Etsy, Shopify, ...). Every other Printify block needs a shop id."
            ),
            categories={BlockCategory.MARKETING, BlockCategory.DATA},
            input_schema=PrintifyListShopsBlock.Input,
            output_schema=PrintifyListShopsBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[("shops", [_TEST_SHOP]), ("shop", _TEST_SHOP)],
            test_mock={"list_shops": lambda credentials: [_TEST_SHOP]},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def list_shops(credentials: PrintifyCredentials) -> list[PrintifyShop]:
        return await PrintifyClient(credentials).list_shops()

    async def run(
        self, input_data: Input, *, credentials: PrintifyCredentials, **kwargs
    ) -> BlockOutput:
        shops = await self.list_shops(credentials)
        yield "shops", shops
        for shop in shops:
            yield "shop", shop
