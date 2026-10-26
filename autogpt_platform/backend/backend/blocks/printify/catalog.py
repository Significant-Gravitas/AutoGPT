from backend.blocks.printify._api import (
    PrintifyBlueprint,
    PrintifyClient,
    PrintifyPrintProvider,
    PrintifyVariant,
)
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
from backend.util.text import singular

MAX_RESULTS = 50

_TEST_BLUEPRINTS = [
    PrintifyBlueprint(id=400, title="Kiss-Cut Stickers", brand="Generic brand"),
    PrintifyBlueprint(id=1268, title="Die-Cut Stickers", brand="Generic brand"),
]
_TEST_PROVIDER = PrintifyPrintProvider(id=99, title="Printify Choice")
_TEST_VARIANTS = [
    PrintifyVariant(id=45748, title='3" × 3" / White', options={"size": '3" × 3"'}),
    PrintifyVariant(id=45750, title='4" × 4" / White', options={"size": '4" × 4"'}),
]


class PrintifySearchCatalogBlock(Block):
    """Find Printify product types (blueprints) by name, e.g. stickers or mugs."""

    class Input(BlockSchemaInput):
        credentials: PrintifyCredentialsInput = printify.credentials_field(
            description="Your Printify personal access token"
        )
        query: str = SchemaField(
            description="Words the product type's name must contain, e.g. 'sticker'"
        )
        limit: int = SchemaField(
            description=f"Most results to return (1-{MAX_RESULTS})",
            default=10,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        blueprints: list[PrintifyBlueprint] = SchemaField(
            description="Matching product types; pass an id to Get Print Options",
            default_factory=list,
        )

    def __init__(self):
        super().__init__(
            id="d45ed8fc-57d0-491a-9124-75777757005b",
            description=(
                "Search the Printify catalog for a product type (a blueprint), "
                "such as kiss-cut stickers, and get its blueprint id."
            ),
            categories={BlockCategory.MARKETING, BlockCategory.SEARCH},
            input_schema=PrintifySearchCatalogBlock.Input,
            output_schema=PrintifySearchCatalogBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "query": "sticker"},
            test_output=[("blueprints", _TEST_BLUEPRINTS)],
            test_mock={"list_blueprints": lambda credentials: _TEST_BLUEPRINTS},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def list_blueprints(
        credentials: PrintifyCredentials,
    ) -> list[PrintifyBlueprint]:
        return await PrintifyClient(credentials).list_blueprints()

    async def run(
        self, input_data: Input, *, credentials: PrintifyCredentials, **kwargs
    ) -> BlockOutput:
        words = input_data.query.lower().split()
        if not words:
            raise ValueError("Say what to search for, e.g. 'sticker'.")
        limit = max(1, min(input_data.limit, MAX_RESULTS))
        blueprints = await self.list_blueprints(credentials)
        yield "blueprints", [b for b in blueprints if _matches(b, words)][:limit]


class PrintifyGetPrintOptionsBlock(Block):
    """The print providers and sizes/colours (variants) for one product type."""

    class Input(BlockSchemaInput):
        credentials: PrintifyCredentialsInput = printify.credentials_field(
            description="Your Printify personal access token"
        )
        blueprint_id: int = SchemaField(description="Product type id from the catalog")
        print_provider_id: int = SchemaField(
            description="Print provider to use; leave 0 for the first one listed",
            default=0,
        )

    class Output(BlockSchemaOutput):
        print_providers: list[PrintifyPrintProvider] = SchemaField(
            description="Every provider that prints this product type",
            default_factory=list,
        )
        print_provider_id: int = SchemaField(description="The provider used below")
        variants: list[PrintifyVariant] = SchemaField(
            description="Sizes and colours this provider offers",
            default_factory=list,
        )
        variant_ids: list[int] = SchemaField(
            description="Every variant id, ready for Create Printify Product",
            default_factory=list,
        )

    def __init__(self):
        super().__init__(
            id="6f07efd0-fdde-4c4c-99f9-7f8fb9827a71",
            description=(
                "List who can print a Printify product type and the variants "
                "(sizes, colours) they offer, with the ids a new product needs."
            ),
            categories={BlockCategory.MARKETING, BlockCategory.DATA},
            input_schema=PrintifyGetPrintOptionsBlock.Input,
            output_schema=PrintifyGetPrintOptionsBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "blueprint_id": 400},
            test_output=[
                ("print_providers", [_TEST_PROVIDER]),
                ("print_provider_id", 99),
                ("variants", _TEST_VARIANTS),
                ("variant_ids", [45748, 45750]),
            ],
            test_mock={
                "list_print_providers": lambda credentials, blueprint_id: [
                    _TEST_PROVIDER
                ],
                "list_variants": lambda credentials, blueprint_id, print_provider_id: (
                    _TEST_VARIANTS
                ),
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def list_print_providers(
        credentials: PrintifyCredentials, blueprint_id: int
    ) -> list[PrintifyPrintProvider]:
        return await PrintifyClient(credentials).list_print_providers(blueprint_id)

    @staticmethod
    async def list_variants(
        credentials: PrintifyCredentials, blueprint_id: int, print_provider_id: int
    ) -> list[PrintifyVariant]:
        return await PrintifyClient(credentials).list_variants(
            blueprint_id, print_provider_id
        )

    async def run(
        self, input_data: Input, *, credentials: PrintifyCredentials, **kwargs
    ) -> BlockOutput:
        providers = await self.list_print_providers(
            credentials, input_data.blueprint_id
        )
        if not providers:
            raise ValueError(
                f"No print provider offers product type {input_data.blueprint_id}."
            )
        provider_id = input_data.print_provider_id or providers[0].id
        variants = await self.list_variants(
            credentials, input_data.blueprint_id, provider_id
        )
        yield "print_providers", providers
        yield "print_provider_id", provider_id
        yield "variants", variants
        yield "variant_ids", [variant.id for variant in variants]


def _matches(blueprint: PrintifyBlueprint, words: list[str]) -> bool:
    text = f"{blueprint.title} {blueprint.brand} {blueprint.model}".lower()
    return all(word in text or singular(word) in text for word in words)
