from backend.blocks.etsy._api import (
    EtsyCategory,
    EtsyClient,
    EtsyProcessingProfile,
    EtsyReturnPolicy,
    EtsyShippingProfile,
    EtsyShop,
)
from backend.blocks.etsy._config import (
    ETSY_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    EtsyCredentials,
    EtsyCredentialsInput,
    etsy,
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

MAX_CATEGORIES = 25

_TEST_SHOP = EtsyShop(
    shop_id=5550123,
    shop_name="StickerStudio",
    url="https://www.etsy.com/shop/StickerStudio",
    currency_code="USD",
)
_TEST_PROFILES = [EtsyShippingProfile(shipping_profile_id=2001, title="Stickers")]
_TEST_PROCESSING = [
    EtsyProcessingProfile(
        readiness_state_id=7001,
        readiness_state="made_to_order",
        processing_days_display_label="1-3 days",
    )
]
_TEST_RETURN_POLICIES = [
    EtsyReturnPolicy(
        return_policy_id=9001,
        accepts_returns=True,
        accepts_exchanges=True,
        return_deadline=30,
    )
]
_TEST_CATEGORIES = [
    EtsyCategory(id=1, name="Paper & Party Supplies", path="Paper & Party Supplies"),
    EtsyCategory(
        id=2078,
        name="Stickers",
        path="Paper & Party Supplies > Paper > Stickers, Labels & Tags > Stickers",
    ),
]


class EtsyGetMyShopBlock(Block):
    """The connected Etsy shop and the shipping profiles new listings can use."""

    class Input(BlockSchemaInput):
        credentials: EtsyCredentialsInput = etsy.credentials_field(
            description="Your Etsy shop"
        )

    class Output(BlockSchemaOutput):
        shop: EtsyShop = SchemaField(description="The connected shop")
        shop_id: int = SchemaField(description="Shop id for the listing blocks")
        shipping_profiles: list[EtsyShippingProfile] = SchemaField(
            description="Shipping profiles; physical listings need one",
            default_factory=list,
        )
        processing_profiles: list[EtsyProcessingProfile] = SchemaField(
            description="Processing times a physical listing can use",
            default_factory=list,
        )
        return_policies: list[EtsyReturnPolicy] = SchemaField(
            description=(
                "Return policies; outside the EU a physical listing needs one "
                "to be published"
            ),
            default_factory=list,
        )

    def __init__(self):
        super().__init__(
            id="11fe7225-3d92-46e7-9ba5-a9bbc5546db3",
            description=(
                "Get your connected Etsy shop, its shop id, and its shipping "
                "profiles, processing profiles and return policies, which the "
                "listing blocks need."
            ),
            categories={BlockCategory.MARKETING, BlockCategory.DATA},
            input_schema=EtsyGetMyShopBlock.Input,
            output_schema=EtsyGetMyShopBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                ("shop", _TEST_SHOP),
                ("shop_id", _TEST_SHOP.shop_id),
                ("shipping_profiles", _TEST_PROFILES),
                ("processing_profiles", _TEST_PROCESSING),
                ("return_policies", _TEST_RETURN_POLICIES),
            ],
            test_mock={
                "get_shop": lambda credentials: _TEST_SHOP,
                "list_shipping_profiles": lambda credentials, shop_id: _TEST_PROFILES,
                "list_processing_profiles": lambda credentials, shop_id: (
                    _TEST_PROCESSING
                ),
                "list_return_policies": lambda credentials, shop_id: (
                    _TEST_RETURN_POLICIES
                ),
            },
            effect=BlockEffect.READ,
            disabled=not ETSY_OAUTH_IS_CONFIGURED,
        )

    @staticmethod
    async def get_shop(credentials: EtsyCredentials) -> EtsyShop:
        return await EtsyClient(credentials).get_my_shop()

    @staticmethod
    async def list_shipping_profiles(
        credentials: EtsyCredentials, shop_id: int
    ) -> list[EtsyShippingProfile]:
        return await EtsyClient(credentials).list_shipping_profiles(shop_id)

    @staticmethod
    async def list_processing_profiles(
        credentials: EtsyCredentials, shop_id: int
    ) -> list[EtsyProcessingProfile]:
        return await EtsyClient(credentials).list_processing_profiles(shop_id)

    @staticmethod
    async def list_return_policies(
        credentials: EtsyCredentials, shop_id: int
    ) -> list[EtsyReturnPolicy]:
        return await EtsyClient(credentials).list_return_policies(shop_id)

    async def run(
        self, input_data: Input, *, credentials: EtsyCredentials, **kwargs
    ) -> BlockOutput:
        shop = await self.get_shop(credentials)
        yield "shop", shop
        yield "shop_id", shop.shop_id
        yield "shipping_profiles", await self.list_shipping_profiles(
            credentials, shop.shop_id
        )
        yield "processing_profiles", await self.list_processing_profiles(
            credentials, shop.shop_id
        )
        yield "return_policies", await self.list_return_policies(
            credentials, shop.shop_id
        )


class EtsySearchCategoriesBlock(Block):
    """Find the Etsy category id a listing needs, by name."""

    class Input(BlockSchemaInput):
        credentials: EtsyCredentialsInput = etsy.credentials_field(
            description="Your Etsy shop"
        )
        query: str = SchemaField(
            description="Category name to look for, e.g. 'stickers'"
        )

    class Output(BlockSchemaOutput):
        categories: list[EtsyCategory] = SchemaField(
            description="Matching categories, the most specific first",
            default_factory=list,
        )

    def __init__(self):
        super().__init__(
            id="cae4a507-99d7-455b-b6f1-4190746e7754",
            description=(
                "Search Etsy's seller categories by name and get the category "
                "(taxonomy) id that Create Etsy Draft Listing needs."
            ),
            categories={BlockCategory.MARKETING, BlockCategory.SEARCH},
            input_schema=EtsySearchCategoriesBlock.Input,
            output_schema=EtsySearchCategoriesBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "query": "stickers"},
            test_output=[("categories", [_TEST_CATEGORIES[1]])],
            test_mock={"list_categories": lambda credentials: _TEST_CATEGORIES},
            effect=BlockEffect.READ,
            disabled=not ETSY_OAUTH_IS_CONFIGURED,
        )

    @staticmethod
    async def list_categories(credentials: EtsyCredentials) -> list[EtsyCategory]:
        return await EtsyClient(credentials).list_categories()

    async def run(
        self, input_data: Input, *, credentials: EtsyCredentials, **kwargs
    ) -> BlockOutput:
        query = input_data.query.strip().lower()
        if not query:
            raise ValueError("Say which category to look for, e.g. 'stickers'.")
        matches = [
            category
            for category in await self.list_categories(credentials)
            if query in category.name.lower()
        ]
        matches.sort(key=lambda category: -category.path.count(">"))
        yield "categories", matches[:MAX_CATEGORIES]
