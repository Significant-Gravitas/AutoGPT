from typing import Any

from backend.blocks.printify._api import PrintifyClient, PrintifyProduct
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
from backend.util.money import price_in_cents

MAX_PRODUCTS = 50

_TEST_PRODUCT = PrintifyProduct(id="prod-sleepy-cat-1", title="Sleepy Cat Sticker")


class PrintifyCreateProductBlock(Block):
    """Create a product in a Printify shop from a design and a product type."""

    class Input(BlockSchemaInput):
        credentials: PrintifyCredentialsInput = printify.credentials_field(
            description="Your Printify personal access token"
        )
        shop_id: int = SchemaField(description="Shop id from List Printify Shops")
        title: str = SchemaField(description="Product title buyers see")
        description: str = SchemaField(description="Product description (HTML allowed)")
        tags: list[str] = SchemaField(
            description="Search tags for the listing", default_factory=list
        )
        blueprint_id: int = SchemaField(description="Product type id from the catalog")
        print_provider_id: int = SchemaField(description="Print provider id")
        variant_ids: list[int] = SchemaField(
            description="Variants (sizes, colours) to sell, from Get Print Options"
        )
        price: float = SchemaField(
            description="Retail price per item in your shop's currency, e.g. 3.99"
        )
        image_id: str = SchemaField(description="Design id from Upload Printify Image")
        print_position: str = SchemaField(
            description="Where the design is printed; stickers and most items use 'front'",
            default="front",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        product_id: str = SchemaField(description="The new product's id")
        product: PrintifyProduct = SchemaField(description="The created product")

    def __init__(self):
        super().__init__(
            id="3caae232-5263-4ab9-a1a1-5d67c7401129",
            description=(
                "Create a Printify product (unpublished) that prints your design "
                "on a product type, with title, description, tags and price."
            ),
            categories={BlockCategory.MARKETING},
            input_schema=PrintifyCreateProductBlock.Input,
            output_schema=PrintifyCreateProductBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "shop_id": 8291,
                "title": "Sleepy Cat Sticker",
                "description": "A vinyl sticker of a very sleepy cat.",
                "tags": ["cat sticker", "kawaii"],
                "blueprint_id": 400,
                "print_provider_id": 99,
                "variant_ids": [45748],
                "price": 3.99,
                "image_id": "img-cat-sticker-1",
            },
            test_output=[
                ("product_id", _TEST_PRODUCT.id),
                ("product", _TEST_PRODUCT),
            ],
            test_mock={"create": lambda credentials, shop_id, product: _TEST_PRODUCT},
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    @staticmethod
    async def create(
        credentials: PrintifyCredentials, shop_id: int, product: dict[str, Any]
    ) -> PrintifyProduct:
        return await PrintifyClient(credentials).create_product(shop_id, product)

    async def run(
        self, input_data: Input, *, credentials: PrintifyCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.variant_ids:
            raise ValueError("Pick at least one variant (size or colour) to sell.")
        product = await self.create(
            credentials, input_data.shop_id, product_body(input_data)
        )
        yield "product_id", product.id
        yield "product", product


def product_body(data: PrintifyCreateProductBlock.Input) -> dict[str, Any]:
    """The create-product request: every variant at one price, one design."""
    cents = price_in_cents(data.price)
    return {
        "title": data.title,
        "description": data.description,
        "tags": data.tags,
        "blueprint_id": data.blueprint_id,
        "print_provider_id": data.print_provider_id,
        "variants": [
            {"id": variant_id, "price": cents, "is_enabled": True}
            for variant_id in data.variant_ids
        ],
        "print_areas": [
            {
                "variant_ids": data.variant_ids,
                "placeholders": [
                    {
                        "position": data.print_position,
                        "images": [
                            {
                                "id": data.image_id,
                                "x": 0.5,
                                "y": 0.5,
                                "scale": 1,
                                "angle": 0,
                            }
                        ],
                    }
                ],
            }
        ],
    }


class PrintifyPublishProductBlock(Block):
    """Publish a Printify product to the store its shop sells through."""

    class Input(BlockSchemaInput):
        credentials: PrintifyCredentialsInput = printify.credentials_field(
            description="Your Printify personal access token"
        )
        shop_id: int = SchemaField(description="Shop id from List Printify Shops")
        product_id: str = SchemaField(description="Product id to publish")

    class Output(BlockSchemaOutput):
        product_id: str = SchemaField(description="The product that is publishing")

    def __init__(self):
        super().__init__(
            id="4b019744-a1d7-49cb-8c54-a04412a70f0f",
            description=(
                "Publish a Printify product to the store connected to its shop "
                "(for example your Etsy shop), making it live for buyers."
            ),
            categories={BlockCategory.MARKETING},
            input_schema=PrintifyPublishProductBlock.Input,
            output_schema=PrintifyPublishProductBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "shop_id": 8291,
                "product_id": _TEST_PRODUCT.id,
            },
            test_output=[("product_id", _TEST_PRODUCT.id)],
            test_mock={"publish": lambda credentials, shop_id, product_id: None},
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def publish(
        credentials: PrintifyCredentials, shop_id: int, product_id: str
    ) -> None:
        await PrintifyClient(credentials).publish_product(shop_id, product_id)

    async def run(
        self, input_data: Input, *, credentials: PrintifyCredentials, **kwargs
    ) -> BlockOutput:
        await self.publish(credentials, input_data.shop_id, input_data.product_id)
        yield "product_id", input_data.product_id


class PrintifyListProductsBlock(Block):
    """List the products in a Printify shop, newest first."""

    class Input(BlockSchemaInput):
        credentials: PrintifyCredentialsInput = printify.credentials_field(
            description="Your Printify personal access token"
        )
        shop_id: int = SchemaField(description="Shop id from List Printify Shops")
        limit: int = SchemaField(
            description=f"Most products to return (1-{MAX_PRODUCTS})", default=20
        )

    class Output(BlockSchemaOutput):
        products: list[PrintifyProduct] = SchemaField(
            description="Products in the shop", default_factory=list
        )

    def __init__(self):
        super().__init__(
            id="3040cada-af4c-4ce6-ac67-41f8820ee727",
            description=(
                "List the products in a Printify shop, with whether each is "
                "visible in the store."
            ),
            categories={BlockCategory.MARKETING, BlockCategory.DATA},
            input_schema=PrintifyListProductsBlock.Input,
            output_schema=PrintifyListProductsBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "shop_id": 8291},
            test_output=[("products", [_TEST_PRODUCT])],
            test_mock={
                "list_products": lambda credentials, shop_id, limit: [_TEST_PRODUCT]
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def list_products(
        credentials: PrintifyCredentials, shop_id: int, limit: int
    ) -> list[PrintifyProduct]:
        return await PrintifyClient(credentials).list_products(shop_id, limit)

    async def run(
        self, input_data: Input, *, credentials: PrintifyCredentials, **kwargs
    ) -> BlockOutput:
        limit = max(1, min(input_data.limit, MAX_PRODUCTS))
        yield "products", await self.list_products(
            credentials, input_data.shop_id, limit
        )
